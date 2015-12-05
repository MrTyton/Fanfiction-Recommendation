import sqlite3
import csv
import sys
import pickle
import evaluator
import argparse
import logging
import os

__all__ = []
__version__ = 0.98
__date__ = '2015-11-20'
__updated__ = '2015-12-03'

def main(argv=None): # IGNORE:C0111
    '''Command line options.'''
    if argv is None:
        argv = sys.argv
    else:
        sys.argv.extend(argv)
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d %I:%M:%S %p', level=logging.INFO)
    program_name = os.path.basename(sys.argv[0])
    program_version = "v%s" % __version__
    program_build_date = str(__updated__)
    #program_version_message = '%%(prog)s %s (%s)' % (program_version, program_build_date)
    program_shortdesc = "Topic modeling with LDA"
    logging.warning('{} {} starting ({}).'.format(program_name, program_version, program_build_date))
    program_license = '''%s

  Created by user_name on %s.
  Copyright 2015 organization_name. All rights reserved.

  Licensed under the Apache License 2.0
  http://www.apache.org/licenses/LICENSE-2.0

  Distributed on an "AS IS" basis without warranties
  or conditions of any kind, either express or implied.

USAGE
''' % (program_shortdesc, str(__date__))
    try:
        # Setup argument parser
        parser = argparse.ArgumentParser(description=program_license, formatter_class=argparse.RawDescriptionHelpFormatter)
        parser.add_argument("-n", "--sample-size", dest="N", default="10000", help="Number of stories to sample for topic modeling [default: %(default)s]")
        parser.add_argument("-a", "--alpha", dest="alpha", default="1.0", help="LDA alpha parameter [default: %(default)s]")
        parser.add_argument("-d", "--dir", dest="basedir", default="/export/apps/dev/fanfiction", help="Directory where to find the database and to write output")
		
        parser.add_argument("-e", "--eta", dest="eta", default=None, help="LDA eta parameter [default: %(default)s]")
        parser.add_argument("-f", "--fold", dest="fold", default="0", help="fold for k-fold xval [default: %(default)s]")
        parser.add_argument("-i", "--num-iter", dest="iter", default="500", help="Number of iterations to use for LDA model fit [default: %(default)s]")
        parser.add_argument("-k", "--num-topics", dest="k", type=int, default="150", help="Number of topics to produce [default: %(default)s]")
        parser.add_argument("-m", "--model-file", dest="modelfile", default=None, help="Number of topics to produce [default: %(default)s]")
        
        args = parser.parse_args()
        '''
        
        '''
        ole = topic.OnlineLDAExperiment(int(args.k), args.basedir)
        if args.modelfile is None:
            modelfile=ole.run_lda_on_summaries(int(args.k), args.alpha, args.eta)
        else:
            modelfile = args.modelfile
        #ole.evaluate_model(modelfile)
        #To evaluate, do something like...
        
        oleg = topic.OnlineLDAExperiment(int(args.k), args.basedir, modelfile=modelfile)
        oleg.prep_for_eval(int(args.fold))
        eval = evaluator.Evaluator(ole, datadir=args.basedir, resultsdir="{}/results".format(args.basedir)) # implements/overrides favorite_likelihood(self, storyID, favorites)
        eval.evaluate()
        #(in parallel)...
        #for each reader (author): 
        #    csv = eval.evaluate_reader(reader)
        
        #for each csv:
        #    results.append(load_csv(csv))
        #compute_statistics(results)

        
    except KeyboardInterrupt:
        ### handle keyboard interrupt ###
        return 0
    except Exception as e:
        logging.error("ERROR: {}".format(e))
        logging.error(traceback.format_exc())
    except: 
        logging.error("Something's wrong.")
    logging.error("GOOD BYE!")
main()
