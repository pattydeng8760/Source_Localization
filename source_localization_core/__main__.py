import sys
from .extract import extract_data, extract_files, extract_surface
from .fft_surface import dft_surface_data, fft_surface_data, source_fft
from .utils import init_logging_from_cut
from .SourceLocalization import SourceLocalization, parse_arguments

def main():
    # Parse CLI args using existing parse_arguments()
    args = parse_arguments()

    if args is None:
        raise RuntimeError("parse_arguments() returned None – please fix its implementation.")

    # Initialize logging once, in serial
    init_logging_from_cut(args.var, args.freq_select)
    
    if args.compute_method == "C":
        from .source_localization_cpp import get_mpi_rank_size_from_env
        rank, _ = get_mpi_rank_size_from_env()
        runner = SourceLocalization(args)
        # if rank == 0:
        #     # initialize the processing only on rank 0
        #     runner.run()
        if args.source_localization:
            runner.load()
            # This calls compute_source_localization_cpp()
            runner.run_source_localization_cpp()
    else:
        # All pre/post processing is serial
        runner = SourceLocalization(args)
        runner.run()  # pre-processing, FFT, etc.

        if args.source_localization:
            # This calls compute_source_localization(), which will
            # choose Python vs C++ kernel based on args.compute_method.
            runner.run_source_localization()


if __name__ == "__main__":
    main()