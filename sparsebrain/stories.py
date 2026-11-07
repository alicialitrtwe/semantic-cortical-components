"""Story list constants for the deep-fMRI dataset.

These lists define the train/test splits used across the project.
The 27-story split follows the idem/compose convention.
"""

# Standard 26 training stories (all subjects have these)
STORIES_TRAIN_26 = [
    "alternateithicatom",
    "souls",
    "avatar",
    "legacy",
    "odetostepfather",
    "undertheinfluence",
    "howtodraw",
    "myfirstdaywiththeyankees",
    "naked",
    "life",
    "stagefright",
    "tildeath",
    "fromboyhoodtofatherhood",
    "sloth",
    "exorcism",
    "haveyoumethimyet",
    "adollshouse",
    "inamoment",
    "theclosetthatateeverything",
    "adventuresinsayingyes",
    "buck",
    "swimmingwithastronauts",
    "thatthingonmyarm",
    "eyespy",
    "itsabox",
    "hangtime",
]

# Standard test story
STORIES_TEST = ["wheretheressmoke"]

# Combined train + test (27 total)
STORIES_27 = STORIES_TRAIN_26 + STORIES_TEST

# 25 stories common to all 8 subjects (excludes "life", missing from UTS04).
# Matches the ridge baseline (scripts/20_fit_ridge_baseline.py).
STORIES_TRAIN_25 = [s for s in STORIES_TRAIN_26 if s != "life"]

# Default training set: 25 common stories for cross-subject comparability.
STORIES_TRAIN = STORIES_TRAIN_25
