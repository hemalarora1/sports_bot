"""Pickleball state machine for the mobile-manipulator Panda (CS225A final project).

High-level layout:
- redis_keys.py     : Redis key profiles for sim (OpenSai) and real robot.
- config.py         : Tunable parameters (court geometry, ready pose, racket geometry, ...).
- ball_tracker.py   : Reads ball position from OpenSai sim or OptiTrack and predicts the intercept.
- swing_planner.py  : Maps an intercept point + return target into pre-swing / strike racket poses.
- pickleball_fsm.py : Main 100 Hz state machine loop.
"""
