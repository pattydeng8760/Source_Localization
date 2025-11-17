from .extract import extract_data, extract_files, extract_surface
from .fft_surface import dft_surface_data,fft_surface_data, source_fft
from .utils import init_logging_from_cut
from .source_localization import SourceLocalization, parse_arguments

def main(args=None):
    # Parse CLI args
    if args is None:
        args = parse_arguments()
    init_logging_from_cut(args.var,args.freq_select)
    runner = SourceLocalization(args)
    runner.run()
    if args.source_localization:
        runner.run_source_localization()

if __name__ == "__main__":
    main()