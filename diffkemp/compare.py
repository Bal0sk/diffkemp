from diffkemp.config import Config
from diffkemp.utils import CMP_OUTPUT_FILE
from diffkemp.llvm_ir.llvm_module import LlvmParam, LlvmModule
from diffkemp.semdiff.caching import SimpLLCache
from diffkemp.semdiff.function_diff import functions_diff
from diffkemp.semdiff.result import Result
from diffkemp.output import YamlOutput
from tempfile import mkdtemp
from timeit import default_timer
import errno
import os
import re
import sys


class OutputDirExistsError(Exception):
    pass


def compare(args):
    """
    Compare two snapshots. Prepares configuration and runs the comparison.
    """
    comparator = SnapshotComparator(args)
    return comparator.run()


class SnapshotComparator:
    """
    Compare the generated snapshots. Runs the semantic comparison and shows
    information about the compared functions that are semantically different.
    """
    MINIMAL_CACHE_FREQ = 2

    def __init__(self, args):
        self.args = args
        self.config = Config.from_args(args)
        self.result = Result(Result.Kind.NONE, args.snapshot_dir_old,
                             args.snapshot_dir_old, start_time=default_timer())
        self.regex_pattern = re.compile(args.regex_filter) \
            if args.regex_filter else None
        self.output_dir = None
        self.writer = self.OutputWriter()

    def run(self):
        # Set the output directory
        try:
            self._set_output_dir()
        except OutputDirExistsError as e:
            sys.stderr.write("{}".format(e))
            sys.exit(errno.EEXIST)

        return self._compare_snapshots()

    def _set_output_dir(self):
        if self.args.stdout:
            return

        if not self.args.output_dir:
            self.output_dir = self._default_output_dir(
                                            self.args.snapshot_dir_old,
                                            self.args.snapshot_dir_new)
            return

        temp_output_dir = self.args.output_dir
        if os.path.isdir(temp_output_dir):
            raise OutputDirExistsError(
                "Error: output directory {} exists\n".format(temp_output_dir))
        self.output_dir = temp_output_dir

    @staticmethod
    def _default_output_dir(src_snapshot, dest_snapshot):
        """Name of the directory to put log files into."""
        base_dirname = "diff-{}-{}".format(
            os.path.basename(os.path.normpath(src_snapshot)),
            os.path.basename(os.path.normpath(dest_snapshot)))
        if os.path.isdir(base_dirname):
            suffix = 0
            dirname = base_dirname
            while os.path.isdir(dirname):
                dirname = "{}-{}".format(base_dirname, suffix)
                suffix += 1
            return dirname
        return base_dirname

    def _compare_snapshots(self):
        for group_name, group in sorted(self.config.snapshot_first.
                                        fun_groups.items()):
            group_printed = False
            group_dir = self._get_group_dir(group_name)
            result_graph = None
            cache = SimpLLCache(mkdtemp())
            module_cache = {}
            modules_to_cache = self._get_modules_to_cache_if_enabled(
                group, group_name)

            for fun, old_fun_desc in sorted(group.functions.items()):
                group_printed, result_graph = self._compare_function(
                                        fun, old_fun_desc, group_name,
                                        group_dir,
                                        group_printed, result_graph, cache,
                                        module_cache, modules_to_cache)
        self.result.graph = result_graph
        self._finalize_output()
        return 0

    def _get_group_dir(self, group_name):
        if self.output_dir is not None and group_name is not None:
            return os.path.join(self.output_dir, group_name)
        return None

    def _get_modules_to_cache_if_enabled(self, group, group_name):
        if self.args.enable_module_cache:
            return self._get_modules_to_cache(
                    group.functions.items(),
                    group_name,
                    self.config.snapshot_second,
                    self.MINIMAL_CACHE_FREQ)
        return set()

    @staticmethod
    def _get_modules_to_cache(functions, group_name, other_snapshot,
                              min_frequency):
        """
        Generates a list of frequently used modules. These will be loaded into
        cache if DiffKemp is running with module caching enable.
        :param functions: List of pairs of functions to be compared along
        with their description objects
        :param group_name: Name of the group the functions are in
        :param other_snapshot: Snapshot object for looking up the functions
        in the other snapshot
        :param min_frequency: Minimal frequency for a module to be included
        into the cache
        :return: Set of modules that should be loaded into module cache
        """
        module_frequency_map = dict()
        for fun, old_fun_desc in functions:
            # Check if the function exists in the other snapshot
            new_fun_desc = other_snapshot.get_by_name(fun, group_name)
            if not new_fun_desc:
                continue
            for fun_desc in [old_fun_desc, new_fun_desc]:
                if not fun_desc.mod:
                    continue
                if fun_desc.mod.llvm not in module_frequency_map:
                    module_frequency_map[fun_desc.mod.llvm] = 0
                module_frequency_map[fun_desc.mod.llvm] += 1
        return {mod for mod, frequency in module_frequency_map.items()
                if frequency >= min_frequency}

    def _compare_function(self, fun, old_fun_desc, group_name, group_dir,
                          group_printed, result_graph, cache,
                          module_cache, modules_to_cache):
        # Check if the function exists in the other snapshot
        new_fun_desc = \
            self.config.snapshot_second.get_by_name(fun, group_name)
        if not new_fun_desc:
            return group_printed, result_graph

        # Check if the module exists in both snapshots
        if not self._modules_exist(old_fun_desc, new_fun_desc):
            return self._handle_missing_module(fun, group_dir,
                                               group_name, old_fun_desc,
                                               group_printed), result_graph
        # If function has a global variable, set it
        glob_var = LlvmParam(old_fun_desc.glob_var) \
            if old_fun_desc.glob_var else None

        # Run the semantic diff
        fun_result = functions_diff(
            mod_first=old_fun_desc.mod, mod_second=new_fun_desc.mod,
            fun_first=fun, fun_second=fun,
            glob_var=glob_var, config=self.config,
            prev_result_graph=result_graph, function_cache=cache,
            module_cache=module_cache,
            modules_to_cache=modules_to_cache)
        result_graph = fun_result.graph

        group_printed = self._handle_fun_result(fun_result, fun,
                                                group_dir,
                                                group_name,
                                                old_fun_desc,
                                                group_printed)
        self._cleanup_modules(old_fun_desc, new_fun_desc)
        return group_printed, result_graph

    @staticmethod
    def _modules_exist(old_fun_desc, new_fun_desc):
        return old_fun_desc.mod is not None and new_fun_desc.mod is not None

    def _handle_missing_module(self, fun, group_dir,
                               group_name, old_fun_desc, group_printed):
        fun_result = Result(Result.Kind.UNKNOWN, fun, fun)
        self.result.add_inner(fun_result)
        group_printed = self._print_fun_result(fun_result, fun,
                                               group_dir,
                                               group_name,
                                               old_fun_desc,
                                               group_printed)
        return group_printed

    def _handle_fun_result(self, fun_result, fun, group_dir,
                           group_name, old_fun_desc, group_printed):

        if self.args.regex_filter is not None:
            # Filter results by regex
            self._filter_result_by_regex(self.regex_pattern,
                                         fun_result)

        self.result.add_inner(fun_result)

        # Printing information about failures and
        # non-equal functions.
        if fun_result.kind in [Result.Kind.NOT_EQUAL,
                               Result.Kind.UNKNOWN,
                               Result.Kind.ERROR] \
           or self.config.full_diff:
            group_printed = self._print_fun_result(fun_result, fun,
                                                   group_dir,
                                                   group_name,
                                                   old_fun_desc,
                                                   group_printed)
        return group_printed

    @staticmethod
    def _filter_result_by_regex(pattern, fun_result):
        for called_res in fun_result.inner.values():
            if pattern.search(called_res.diff):
                return
        fun_result.kind = Result.Kind.EQUAL

    def _print_fun_result(self, fun_result, fun, group_dir,
                          group_name, old_fun_desc, group_printed):
        if fun_result.kind != Result.Kind.NOT_EQUAL and \
           not self.config.full_diff:
            # Print the group name if needed
            if group_name is not None and not group_printed:
                print("{}:".format(group_name))
                group_printed = True
            print("{}: {}".format(fun, str(fun_result.kind)))
            return group_printed

        # Create the output directory if needed
        if self.output_dir is not None:
            self._ensure_dir_exists(self.output_dir)

        # Create the group directory or print the group name
        # if needed
        if group_dir is not None:
            self._ensure_dir_exists(group_dir)
        elif group_name is not None and not group_printed:
            print("{}:".format(group_name))
            group_printed = True

        self.print_syntax_diff(
            fun=fun,
            fun_result=fun_result,
            fun_tag=old_fun_desc.tag,
            output_dir=group_dir if group_dir else self.output_dir,
            initial_indent=2 if (group_name is not None and
                                 group_dir is None) else 0)

        return group_printed

    def _ensure_dir_exists(self, path):
        if path and not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

    def print_syntax_diff(self, fun, fun_result, fun_tag, output_dir,
                          initial_indent):
        """
        Log syntax diff of 2 functions. If log_files is set, the output is
        printed into a separate file, otherwise it goes to stdout.
        :param fun: Name of the analysed function
        :param fun_tag: Analysed function tag
        :param fun_result: Result of the analysis
        :param output_dir: True if the output is to be written into a file
        :param initial_indent: Initial indentation of printed messages
        """

        old_dir_abs_path = os.path.join(
            os.path.abspath(self.args.snapshot_dir_old), "")
        new_dir_abs_path = os.path.join(
            os.path.abspath(self.args.snapshot_dir_new), "")

        if fun_result.kind == Result.Kind.NOT_EQUAL or (
                self.config.full_diff and
                any([x.diff for x in fun_result.inner.values()])):

            # sets the output and indent in the writer class
            self.writer._open_output_file_set_indent(
                output_dir, fun, fun_tag, initial_indent)
            for called_res in sorted(fun_result.inner.values(),
                                     key=lambda r: r.first.name):
                if called_res.diff == "" and called_res.first.covered:
                    # Do not print empty diffs
                    continue
                self.writer.write_called_result(called_res,
                                                self.config.show_diff,
                                                self.args.snapshot_dir_old,
                                                self.args.snapshot_dir_new,
                                                old_dir_abs_path,
                                                new_dir_abs_path,
                                                output_dir)
            self.writer.close_file()

    @staticmethod
    def _cleanup_modules(old_fun_desc, new_fun_desc):
        # Clean LLVM modules (allow GC to collect the occupied memory)
        old_fun_desc.mod.clean_module()
        new_fun_desc.mod.clean_module()
        LlvmModule.clean_all()

    def _create_yaml_output(self):
        old_dir_abs = \
            os.path.join(os.path.abspath(self.args.snapshot_dir_old), "")
        new_dir_abs = \
            os.path.join(os.path.abspath(self.args.snapshot_dir_new), "")
        yaml_output = YamlOutput(snapshot_dir_old=old_dir_abs,
                                 snapshot_dir_new=new_dir_abs,
                                 result=self.result)
        yaml_output.save(output_dir=self.output_dir, file_name=CMP_OUTPUT_FILE)

    def _print_stats(self, errors, extended_stat):
        print(f"\nStatistics\n{'-' * 11}")
        self.result.stop_time = default_timer()
        self.result.report_stat(errors, extended_stat)

    class OutputWriter:
        """
        Handles formatted output of function diffs, including indentation
        and callstack printing, to either a file or stdout.
        """
        def __init__(self):
            self.output = None
            self.indent = 0

        @staticmethod
        def _text_indent(text, width):
            """
            Indent each line in the text by a number of spaces given by width
            """
            return ''.join(" "*width + line for line in text.splitlines(True))

        def _open_output_file_set_indent(self, output_dir, fun,
                                         fun_tag, initial_indent):
            if output_dir:
                output = open(os.path.join(output_dir, "{}.diff".format(fun)),
                              "w")
                output.write(
                    "Found differences in functions called by {}".format(fun))
                if fun_tag is not None:
                    output.write(" ({})".format(fun_tag))
                output.write("\n\n")
                indent = initial_indent + 2
            else:
                output = sys.stdout
                if fun_tag is not None:
                    output.write(
                        self._text_indent("{} ({}):\n".format(fun, fun_tag),
                                          initial_indent))
                else:
                    output.write(self._text_indent("{}:\n".format(fun),
                                 initial_indent))
                indent = initial_indent + 4

            self.output = output
            self.indent = indent

        def write_called_result(self, called_res,
                                show_diff, snapshot_dir_old,
                                snapshot_dir_new, old_dir_abs_path,
                                new_dir_abs_path, output_dir):
            self.output.write(
                self._text_indent(
                    "{} differs:\n".format(called_res.first.name),
                    self.indent - 2))

            if not output_dir:
                self.output.write(self._text_indent("{{{\n", self.indent - 2))

            if called_res.first.callstack:
                self._write_callstack(called_res.first.callstack,
                                      snapshot_dir_old, old_dir_abs_path)
            if called_res.second.callstack:
                self._write_callstack(called_res.second.callstack,
                                      snapshot_dir_new, new_dir_abs_path)
            if show_diff:
                self._write_diff(called_res)
            if not output_dir:
                self.output.write(self._text_indent("}}}\n", self.indent - 2))
            self.output.write("\n")

        def _write_callstack(self, callstack, label, abs_path):
            self.output.write(
                self._text_indent("Callstack ({}):\n".format(label),
                                  self.indent))
            self.output.write(
                self._text_indent(callstack.as_str_with_rel_paths(abs_path),
                                  self.indent))
            self.output.write("\n\n")

        def _write_diff(self, called_res):
            if (called_res.diff.strip() == "" and
                    called_res.macro_diff is not None):
                self.output.write(self._text_indent(
                    "\n".join(map(str, called_res.macro_diff)),
                    self.indent))
            else:
                self.output.write(self._text_indent("Diff:\n", self.indent))
                self.output.write(
                    self._text_indent(called_res.diff, self.indent))

        def close_file(self):
            if self.output is not None and self.output is not sys.stdout:
                self.output.close()
            self.output = None

    def _finalize_output(self):
        # Create yaml output
        if self.output_dir is not None and os.path.isdir(self.output_dir):
            self._create_yaml_output()
        self.config.snapshot_first.finalize()
        self.config.snapshot_second.finalize()

        if self.output_dir is not None and os.path.isdir(self.output_dir):
            print("Differences stored in {}/".format(self.output_dir))
        if self.args.report_stat or self.args.extended_stat:
            self._print_stats(self.args.show_errors,
                              self.args.extended_stat)
