Nyctolith data
==============
patches.npy  : uint8 array, shape (12000, 100, 100). Row index == tile_uid.
               Each row is one anonymized 100x100 daytime lunar grayscale image
               patch (100 m/pixel, 10 km x 10 km), randomly D4-oriented.
train.csv    : tile_uid, rock_abundance  (labeled training patches)
test.csv     : tile_uid                  (patches to score; predict each)
sample_submission.csv : required output format.
Submit working/submission.csv with columns [tile_uid, rock_abundance] for every
test tile_uid: a single non-negative rock-abundance estimate (areal fraction) per patch.
