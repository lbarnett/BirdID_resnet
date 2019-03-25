"""Split images into training, validation, and test folders."""
from __future__ import print_function
import argparse
import os
import shutil
from utils import imgloader as il
from const import TRAIN_MODE, VAL_MODE, TEST_MODE
# import collections


def parse_filename(fn):
    """
    Given a filename, break out the date/time components.

    Given a filename of the form 'yyyy-mm-dd_hh.mm.ss.ext', return
    the date as a string and the time as number of seconds since midnight.

    args:
    fn - filename in the described format

    returns:
    date - a string containing the date component of the filename
    secs - the time in seconds since midnight

    """
    date, timestr = fn.split('_')
    # print('Date: ' + date + '   timestr: ' + timestr)
    time_parts = timestr.split('.')
    # Handle situation where two images were captured during the same sec.
    time_parts[2] = time_parts[2].split('-')[0]
    sec = int(time_parts[0]) * 3600 + int(time_parts[1])*60 + int(time_parts[2])

    return date, sec


def what_next(train_frac, val_frac, test_frac, total_processed, counts):
    """
    Decide where the next run of image files should be written.

    Choose where the next run of images goes (train, validate, or test)
    based on percentages already copied and the target percentages in
    each of those three subdirectories.
    :rtype: int
    """
    if counts[TRAIN_MODE]/(total_processed * 1.0) < train_frac:
        curr_mode = TRAIN_MODE
    elif counts[VAL_MODE]/(total_processed * 1.0) < val_frac or \
            test_frac == 0.0:
        curr_mode = VAL_MODE
    else:
        curr_mode = TEST_MODE

    return curr_mode


def main():
    """Split images, taking temporal locality into account."""
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir',
                        help='Path name for classified image set')
    parser.add_argument('split_dir',
                        help='Path for base folder of train/val/test imgs')
    parser.add_argument('train_fraction',
                        help='Fraction of images to add to training set')
    parser.add_argument('validate_fraction',
                        help='Fraction of images to add to validation set')
    parser.add_argument('test_fraction',
                        help='Fraction of images to add to test set (0 ok)')
    parser.add_argument('min_gap',
                        default='45'
                        help='Minimum gap separating runs in seconds')

    args = parser.parse_args()

    print('Splitting ' + args.source_dir + '\n' +
          '(test/val/train):  (' + args.train_fraction + '/' +
          args.validate_fraction + '/' + args.test_fraction + ')\n')

    train_frac = float(args.train_fraction)
    val_frac = float(args.validate_fraction)
    test_frac = float(args.test_fraction)

    # Fetch the list of paths to files in the source directory.
    filepaths, categories, label_names = \
        il.get_paths_with_labels(args.source_dir)

    # Need filepaths sorted by the filename within the folder, which
    # get_paths_with_labels doesn't guarantee.
    decorated = [(filepaths[i], categories[i]) for i in xrange(len(filepaths))]
    date_ordered = sorted(decorated)
    filepaths = [x[0] for x in date_ordered]
    categories = [x[1] for x in date_ordered]

    # Figure out how many files in each category - we need this to
    # be sure we're getting the proportions right.
    # Maybe didn't really need this...
    # counter = collections.Counter(categories)
    # cat_sizes = counter.values()

    # Debugging
    # print('Category sizes')
    # for i in xrange(len(cat_sizes)):
    #     print(label_names[i] + ': ' + str(cat_sizes[i]))

    # print('Path\t\t\tCategory\t\tLabel name')
    # for i in xrange(len(filepaths)):
    #    print(filepaths[i] + '\t' + str(categories[i]) + '\t' +
    #          label_names[categories[i]])

    sub_dirs = ['train', 'val', 'test']
    run_separation_threshold = int(args.min_gap)  # seconds

    if not os.path.exists(args.split_dir):
        os.makedirs(args.split_dir)

    train_path = os.path.join(args.split_dir, 'train')
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    val_path = os.path.join(args.split_dir, 'val')
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    idx = 0

    # Loop as long as we have image files left
    while idx < len(filepaths):
        # Set up to handle all the file within one category
        counts = [0, 0, 0]
        processed_in_current_cat = 0
        curr_cat = categories[idx]

        # Create subdirs for this category in train/val/test
        # directories
        train_cat_path = os.path.join(args.split_dir, 'train',
                                      label_names[curr_cat])
        val_cat_path = os.path.join(args.split_dir, 'val',
                                    label_names[curr_cat])
        test_cat_path = os.path.join(args.split_dir, 'test',
                                     label_names[curr_cat])
        # Be sure category folders exist for all three subtrees
        if not os.path.exists(train_cat_path):
            os.makedirs(train_cat_path)
        if not os.path.exists(val_cat_path):
            os.makedirs(val_cat_path)
        if not os.path.exists(test_cat_path):
            os.makedirs(test_cat_path)

        # Loop through all images in category curr_cat and distribute
        # among train, validate, and test according to requested percentages
        # Keep multiple images that are part of one visit to the feeder
        # as indicated by the timestamp together. The end percentages will
        # be approximate.
        while idx < len(filepaths) and categories[idx] == curr_cat:
            # At the beginning of a run of image files.
            # There must be at least one file in the current category.
            # Put it somewhere and use it as the basis for the current run.
            # Otherwise, decide which set to assign this run to.
            if processed_in_current_cat == 0:
                curr_mode = TRAIN_MODE
                src_file_name = os.path.basename(filepaths[idx])
                file_path = os.path.join(args.split_dir, sub_dirs[curr_mode],
                                         label_names[curr_cat], src_file_name)
                shutil.copy(filepaths[idx], file_path)
                processed_in_current_cat = 1
                counts[curr_mode] = 1
                prev_date, prev_sec = parse_filename(src_file_name)
                idx = idx + 1
            else:
                curr_mode = what_next(train_frac, val_frac, test_frac,
                                      processed_in_current_cat, counts)

            # Process this run
            while idx < len(filepaths) and categories[idx] == curr_cat:
                src_file_name = os.path.basename(filepaths[idx])
                curr_date, curr_sec = parse_filename(src_file_name)
                # debugging
                # print ('src_file_name: %s  curr_date: %s  curr_sec: %d' %
                #       (src_file_name, curr_date, curr_sec))
                # print ('\t\t\tprev_date: %s   prev_sec: %d' %
                #       (prev_date, prev_sec))

                if curr_date != prev_date or \
                        curr_sec - prev_sec >= run_separation_threshold:
                    # Run is over, set up for next one
                    curr_mode = what_next(train_frac, val_frac, test_frac,
                                          processed_in_current_cat, counts)

                file_path = os.path.join(args.split_dir,
                                         sub_dirs[curr_mode],
                                         label_names[curr_cat],
                                         src_file_name)
                shutil.copy(filepaths[idx], file_path)
                processed_in_current_cat = processed_in_current_cat + 1
                counts[curr_mode] = counts[curr_mode] + 1
                prev_date = curr_date
                prev_sec = curr_sec
                idx = idx + 1

        # Category is done, set up for processing the next category
        if idx < len(filepaths):
            print('Finished category: ' + label_names[curr_cat] +
                  ' train:' + str(counts[TRAIN_MODE]) + '(' +
                  str(counts[TRAIN_MODE]/(processed_in_current_cat * 1.0)) +
                  ')' +
                  ' val: ' + str(counts[VAL_MODE]) + '(' +
                  str(counts[VAL_MODE]/(processed_in_current_cat * 1.0)) +
                  ')' +
                  ' test: ' + str(counts[TEST_MODE]) + '(' +
                  str(counts[TEST_MODE]/(processed_in_current_cat*1.0)) + ')')

            curr_cat = categories[idx]
            processed_in_current_cat = 0
            counts = [0, 0, 0]


if __name__ == '__main__':
    main()
