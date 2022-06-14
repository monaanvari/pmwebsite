"""
Set of macros to use globally for iGibson
"""

# Whether to generate a headless or non-headless application upon iGibson startup
HEADLESS = False

# Whether to use extra settings (verboseness, extra GUI features) for debugging
DEBUG = True

# Whether to enable (a) [global / robot] contact checking or not
# Note: You can enable the robot contact checking, even if global checking is disabled
# If global checking is enabled but robot checking disabled, global checking will take
# precedence (i.e.: robot will still have contact checking)
# TODO: Remove this once we have an optimized solution
ENABLE_GLOBAL_CONTACT_REPORTING = False
ENABLE_ROBOT_CONTACT_REPORTING = True

# Whether to use omni's particles feature (e.g. for fluids) or not
# This also dictates whether we need to use GPU dynamics or not
ENABLE_OMNI_PARTICLES = False

# Whether to use omni's flatcache feature or not (can speed up simulation)
ENABLE_FLATCACHE = False

# Whether to use continuous collision detection or not (slower simulation, but can prevent
# objects from tunneling through each other)
ENABLE_CCD = False

# Aggregate pairs setting -- default is 1024, but is often insufficient for large scenes
GPU_PAIRS_CAPACITY = 32 * 1024

# Whether the public version of IsaacSim is being used
# TODO: Remove this once we unify omni version being used
IS_PUBLIC_ISAACSIM = True

# (Demo-purpose) Whether to activate Assistive Grasping mode for Cloth (it's handled differently from RigidBody)
AG_CLOTH = False