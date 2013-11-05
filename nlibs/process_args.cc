#include "process_args.h"

Options options;

int process_args(int argc, char **argv) {
  int c;
  options.inputFileName[0] = '\n';
  while (1) {
    static struct option long_options[] = {
      /* These options set a flag. */
      {"calcChange", no_argument, 0, 'c'},
      /* These options don't set a flag.
         We distinguish them by their indices. */
      {"input",  required_argument, 0, 'i'},
      {"maxIters",  required_argument, 0, 'm'},
      {"stats", no_argument, 0, 's'},
      {"help",   no_argument, 0, 'h'},
      {0, 0, 0, 0}
    };
    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long (argc, argv, "cim:sh",
        long_options, &option_index);
    /* Detect the end of the options. */
    if (c == -1)
      break;

    switch (c) {
      case 0:
        /* If this option set a flag, do nothing else now. */
        if (long_options[option_index].flag != 0)
          break;
        printf ("option %s", long_options[option_index].name);
        if (optarg)
          printf (" with arg %s", optarg);
        printf ("\n");
        break;

      case 'c':
        options.calcChange = true;
        break;
      case 's':
        options.stats = true;
        break;
      case 'i':
        strcpy(options.inputFileName, optarg);
        break;
      case 'm':
        options.maxIters = atol(optarg);
        break;
      case 'h':
        break;
      case '?':
        /* getopt_long already printed an error message. */
        break;

      default:
        abort ();
    }
  }

  /* Print any remaining command line arguments (not options). */
  if (optind < argc)
  {
    printf ("non-option ARGV-elements: ");
    while (optind < argc)
      printf ("%s ", argv[optind++]);
    putchar ('\n');
  }
  return 0;
}

void print_args() {
  printf("{\t");
  printf("calcChange= %s\t", options.calcChange ? "true" : "false");
  printf("stats= %s\t", options.stats ? "true" : "false");
  printf("inputFileName= %s\t", options.inputFileName);
  printf("maxIters= %d\t", options.maxIters);
  printf("}\n");
}
