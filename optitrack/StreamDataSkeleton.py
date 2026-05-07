#Copyright © 2018 Naturalpoint
#
#Licensed under the Apache License, Version 2.0 (the "License")
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.


# OptiTrack NatNet direct depacketization sample for Python 3.x
#
# Uses the Python NatNetClient.py library to establish a connection (by creating a NatNetClient),
# and receive data via a NatNet connection and decode it using the NatNetClient library.

import json
import os
import sys
import time
from NatNetClient import NatNetClient
import DataDescriptions
import MoCapData
import numpy as np
import redis
import signal
import sys

is_looping = True
def signal_handler(sig, frame):
    is_looping = False
    print('You pressed Ctrl+C! Terminating program')
    sys.exit(0)

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

# create redis pipeline
pipeline = redis_client.pipeline()
n_skeletons = 2
min_id = 1
max_id = 51
indices = [5, 3, 2, 47, 50, 23, 32, 8, 27, 49, 45]


# ---------- World-frame calibration -------------------------------------------
#
# OptiTrack publishes positions in its own room frame. The pickleball FSM and
# the C++ controllers use a "world" frame anchored at the robot base (+X
# forward, +Y lateral, +Z up; see sports_bot/state_machine/config.py).
#
# We apply a calibration transform T_world_optitrack to every rigid body before
# publishing. The transform can be supplied via a JSON file next to this script
# (`world_calibration.json`); if missing, identity is used and the OptiTrack
# frame is treated as the world frame.
#
# Expected JSON schema (any field omitted falls back to identity):
#   {
#     "translation": [tx, ty, tz],
#     "rotation":    [[r00, r01, r02],
#                     [r10, r11, r12],
#                     [r20, r21, r22]]
#   }
# t_world_p = R * t_optitrack_p + translation.

_CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "world_calibration.json")

def _load_world_calibration(path):
    R = np.eye(3)
    t = np.zeros(3)
    if not os.path.isfile(path):
        print(f"[optitrack] no calibration file at {path}, using identity")
        return R, t
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if "rotation" in data:
            R = np.array(data["rotation"], dtype=float)
            assert R.shape == (3, 3), f"rotation must be 3x3, got {R.shape}"
        if "translation" in data:
            t = np.array(data["translation"], dtype=float)
            assert t.shape == (3,), f"translation must be 3-vec, got {t.shape}"
        print(f"[optitrack] loaded world calibration from {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[optitrack] failed to read {path}: {exc}; using identity")
        R, t = np.eye(3), np.zeros(3)
    return R, t

R_WORLD_OPTI, T_WORLD_OPTI = _load_world_calibration(_CALIBRATION_FILE)

def _opti_to_world_position(pos):
    p = np.asarray(pos, dtype=float)
    return (R_WORLD_OPTI @ p) + T_WORLD_OPTI

def _opti_to_world_quat(rot):
    # Motive streams quaternion as (qx, qy, qz, qw). We rotate by R_WORLD_OPTI,
    # which is q_world = q_offset * q_optitrack. For now we publish the raw
    # quaternion alongside the rotated one; the FSM only uses position.
    return rot

# This is a callback function that gets connected to the NatNet client
# and called once per mocap frame.
def receive_new_frame(data_dict):
    order_list=[ "frameNumber", "markerSetCount", "unlabeledMarkersCount", "rigidBodyCount", "skeletonCount",
                "labeledMarkerCount", "timecode", "timecodeSub", "timestamp", "isRecording", "trackedModelsChanged" ]
    dump_args = False
    if dump_args == True:
        out_string = "    "
        for key in data_dict:
            out_string += key + "="
            if key in data_dict :
                out_string += data_dict[key] + " "
            out_string+="/"
        print(out_string)

RIGID_BODY_POS_KEY = "sai2::optitrack::rigid_body_pos::"
RIGID_BODY_ORI_KEY = "sai2::optitrack::rigid_body_ori::"
RIGID_BODY_RAW_POS_KEY = "sai2::optitrack::raw::rigid_body_pos::"
RIGID_BODY_RAW_ORI_KEY = "sai2::optitrack::raw::rigid_body_ori::"
def receive_rigid_body_frame( new_id, position, rotation ):
    # Publish raw OptiTrack values (room frame) for debugging.
    redis_client.set(RIGID_BODY_RAW_POS_KEY + str(new_id),
                     json.dumps([float(x) for x in position]))
    redis_client.set(RIGID_BODY_RAW_ORI_KEY + str(new_id),
                     json.dumps([float(x) for x in rotation]))
    # Publish world-frame (calibrated) values that the FSM / controller consume.
    world_pos = _opti_to_world_position(position)
    world_rot = _opti_to_world_quat(rotation)
    redis_client.set(RIGID_BODY_POS_KEY + str(new_id),
                     json.dumps(world_pos.tolist()))
    redis_client.set(RIGID_BODY_ORI_KEY + str(new_id),
                     json.dumps([float(x) for x in world_rot]))

def receive_skeleton_frame(new_id, skeleton):
    
    # # Do timing 
    # start_time = time.time()
    
    # Iterate over each rigid body in the skeleton's rigid body list
    for i, rigid_body in enumerate(skeleton.rigid_body_list):
        # if i in indices:
        # Construct Redis keys for position and orientation
        position_key = f"{new_id}::{i + 1}::pos"
        orientation_key = f"{new_id}::{i + 1}::ori"

        # Convert position and orientation to string format
        position_str = '[' + ', '.join(map(str, rigid_body.pos)) + ']'
        orientation_str = '[' + ', '.join(map(str, rigid_body.rot)) + ']'

        # Set the position and orientation in Redis
        # redis_client.set(position_key, position_str)
        # redis_client.set(orientation_key, orientation_str)
        
        pipeline.set(position_key, position_str)  # change to pipeline 
        pipeline.set(orientation_key, orientation_str)
        
    pipeline.execute()


def add_lists(totals, totals_tmp):
    totals[0]+=totals_tmp[0]
    totals[1]+=totals_tmp[1]
    totals[2]+=totals_tmp[2]
    return totals

def print_configuration(natnet_client):
    natnet_client.refresh_configuration()
    print("Connection Configuration:")
    print("  Client:          %s"% natnet_client.local_ip_address)
    print("  Server:          %s"% natnet_client.server_ip_address)
    print("  Command Port:    %d"% natnet_client.command_port)
    print("  Data Port:       %d"% natnet_client.data_port)

    changeBitstreamString = "  Can Change Bitstream Version = "
    if natnet_client.use_multicast:
        print("  Using Multicast")
        print("  Multicast Group: %s"% natnet_client.multicast_address)
        changeBitstreamString+="false"
    else:
        print("  Using Unicast")
        changeBitstreamString+="true"

    #NatNet Server Info
    application_name = natnet_client.get_application_name()
    nat_net_requested_version = natnet_client.get_nat_net_requested_version()
    nat_net_version_server = natnet_client.get_nat_net_version_server()
    server_version = natnet_client.get_server_version()

    print("  NatNet Server Info")
    print("    Application Name %s" %(application_name))
    print("    MotiveVersion  %d %d %d %d"% (server_version[0], server_version[1], server_version[2], server_version[3]))
    print("    NatNetVersion  %d %d %d %d"% (nat_net_version_server[0], nat_net_version_server[1], nat_net_version_server[2], nat_net_version_server[3]))
    print("  NatNet Bitstream Requested")
    print("    NatNetVersion  %d %d %d %d"% (nat_net_requested_version[0], nat_net_requested_version[1],\
       nat_net_requested_version[2], nat_net_requested_version[3]))

    print(changeBitstreamString)
    #print("command_socket = %s"%(str(natnet_client.command_socket)))
    #print("data_socket    = %s"%(str(natnet_client.data_socket)))
    print("  PythonVersion    %s"%(sys.version))


def print_commands(can_change_bitstream):
    outstring = "Commands:\n"
    outstring += "Return Data from Motive\n"
    outstring += "  s  send data descriptions\n"
    outstring += "  r  resume/start frame playback\n"
    outstring += "  p  pause frame playback\n"
    outstring += "     pause may require several seconds\n"
    outstring += "     depending on the frame data size\n"
    outstring += "Change Working Range\n"
    outstring += "  o  reset Working Range to: start/current/end frame 0/0/end of take\n"
    outstring += "  w  set Working Range to: start/current/end frame 1/100/1500\n"
    outstring += "Return Data Display Modes\n"
    outstring += "  j  print_level = 0 supress data description and mocap frame data\n"
    outstring += "  k  print_level = 1 show data description and mocap frame data\n"
    outstring += "  l  print_level = 20 show data description and every 20th mocap frame data\n"
    outstring += "Change NatNet data stream version (Unicast only)\n"
    outstring += "  3  Request NatNet 3.1 data stream (Unicast only)\n"
    outstring += "  4  Request NatNet 4.1 data stream (Unicast only)\n"
    outstring += "General\n"
    outstring += "  t  data structures self test (no motive/server interaction)\n"
    outstring += "  c  print configuration\n"
    outstring += "  h  print commands\n"
    outstring += "  q  quit\n"
    outstring += "\n"
    outstring += "NOTE: Motive frame playback will respond differently in\n"
    outstring += "       Endpoint, Loop, and Bounce playback modes.\n"
    outstring += "\n"
    outstring += "EXAMPLE: PacketClient [serverIP [ clientIP [ Multicast/Unicast]]]\n"
    outstring += "         PacketClient \"192.168.10.14\" \"192.168.10.14\" Multicast\n"
    outstring += "         PacketClient \"127.0.0.1\" \"127.0.0.1\" u\n"
    outstring += "\n"
    print(outstring)

def request_data_descriptions(s_client):
    # Request the model definitions
    s_client.send_request(s_client.command_socket, s_client.NAT_REQUEST_MODELDEF,    "",  (s_client.server_ip_address, s_client.command_port) )

def test_classes():
    totals = [0,0,0]
    print("Test Data Description Classes")
    totals_tmp = DataDescriptions.test_all()
    totals=add_lists(totals, totals_tmp)
    print("")
    print("Test MoCap Frame Classes")
    totals_tmp = MoCapData.test_all()
    totals=add_lists(totals, totals_tmp)
    print("")
    print("All Tests totals")
    print("--------------------")
    print("[PASS] Count = %3.1d"%totals[0])
    print("[FAIL] Count = %3.1d"%totals[1])
    print("[SKIP] Count = %3.1d"%totals[2])

def my_parse_args(arg_list, args_dict):
    # set up base values
    arg_list_len=len(arg_list)
    if arg_list_len>1:
        args_dict["serverAddress"] = arg_list[1]
        if arg_list_len>2:
            args_dict["clientAddress"] = arg_list[2]
        if arg_list_len>3:
            if len(arg_list[3]):
                args_dict["use_multicast"] = True
                if arg_list[3][0].upper() == "U":
                    args_dict["use_multicast"] = False

    return args_dict


if __name__ == "__main__":
    
    signal.signal(signal.SIGINT, signal_handler)

    optionsDict = {}
    optionsDict["clientAddress"] = "172.24.68.64"
    optionsDict["serverAddress"] = "172.24.68.48"
    optionsDict["use_multicast"] = False

    # This will create a new NatNet client
    optionsDict = my_parse_args(sys.argv, optionsDict)

    streaming_client = NatNetClient()
    streaming_client.set_client_address(optionsDict["clientAddress"])
    streaming_client.set_server_address(optionsDict["serverAddress"])
    streaming_client.set_use_multicast(optionsDict["use_multicast"])

    # Configure the streaming client to call our rigid body handler on the emulator to send data out.
    streaming_client.new_frame_listener = receive_new_frame
    streaming_client.rigid_body_listener = receive_rigid_body_frame
    streaming_client.skeleton_listener = receive_skeleton_frame
    
    # Set print level
    streaming_client.set_print_level(0)

    # Start up the streaming client now that the callbacks are set up.
    # This will run perpetually, and operate on a separate thread.
    is_running = streaming_client.run()
    if not is_running:
        print("ERROR: Could not start streaming client.")
        try:
            sys.exit(1)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    is_looping = True
    time.sleep(1)
    if streaming_client.connected() is False:
        print("ERROR: Could not connect properly.  Check that Motive streaming is on.")
        try:
            sys.exit(2)
        except SystemExit:
            print("...")
        finally:
            print("exiting")

    # print_configuration(streaming_client)
    # print("\n")
    # print_commands(streaming_client.can_change_bitstream_version())

    while is_looping:
        time.sleep(1)

    # while is_looping:
    #     inchars = input('Enter command or (\'h\' for list of commands)\n')
    #     if len(inchars)>0:
    #         c1 = inchars[0].lower()
    #         if c1 == 'h' :
    #             print_commands(streaming_client.can_change_bitstream_version())
    #         elif c1 == 'c' :
    #             print_configuration(streaming_client)
    #         elif c1 == 's':
    #             request_data_descriptions(streaming_client)
    #             time.sleep(1)
    #         elif (c1 == '3') or (c1 == '4'):
    #             if streaming_client.can_change_bitstream_version():
    #                 tmp_major = 4
    #                 tmp_minor = 1
    #                 if(c1 == '3'):
    #                     tmp_major = 3
    #                     tmp_minor = 1
    #                 return_code = streaming_client.set_nat_net_version(tmp_major,tmp_minor)
    #                 time.sleep(1)
    #                 if return_code == -1:
    #                     print("Could not change bitstream version to %d.%d"%(tmp_major,tmp_minor))
    #                 else:
    #                     print("Bitstream version at %d.%d"%(tmp_major,tmp_minor))
    #             else:
    #                 print("Can only change bitstream in Unicast Mode")

    #         elif c1 == 'p':
    #             sz_command="TimelineStop"
    #             return_code = streaming_client.send_command(sz_command)
    #             time.sleep(1)
    #             print("Command: %s - return_code: %d"% (sz_command, return_code) )
    #         elif c1 == 'r':
    #             sz_command="TimelinePlay"
    #             return_code = streaming_client.send_command(sz_command)
    #             print("Command: %s - return_code: %d"% (sz_command, return_code) )
    #         elif c1 == 'o':
    #             tmpCommands=["TimelinePlay",
    #                         "TimelineStop",
    #                         "SetPlaybackStartFrame,0",
    #                         "SetPlaybackStopFrame,1000000",
    #                         "SetPlaybackLooping,0",
    #                         "SetPlaybackCurrentFrame,0",
    #                         "TimelineStop"]
    #             for sz_command in tmpCommands:
    #                 return_code = streaming_client.send_command(sz_command)
    #                 print("Command: %s - return_code: %d"% (sz_command, return_code) )
    #             time.sleep(1)
    #         elif c1 == 'w':
    #             tmp_commands=["TimelinePlay",
    #                         "TimelineStop",
    #                         "SetPlaybackStartFrame,1",
    #                         "SetPlaybackStopFrame,1500",
    #                         "SetPlaybackLooping,0",
    #                         "SetPlaybackCurrentFrame,100",
    #                         "TimelineStop"]
    #             for sz_command in tmp_commands:
    #                 return_code = streaming_client.send_command(sz_command)
    #                 print("Command: %s - return_code: %d"% (sz_command, return_code) )
    #             time.sleep(1)
    #         elif c1 == 't':
    #             test_classes()

    #         elif c1 == 'j':
    #             streaming_client.set_print_level(0)
    #             print("Showing only received frame numbers and supressing data descriptions")
    #         elif c1 == 'k':
    #             streaming_client.set_print_level(1)
    #             print("Showing every received frame")

    #         elif c1 == 'l':
    #             print_level = streaming_client.set_print_level(20)
    #             print_level_mod = print_level % 100
    #             if(print_level == 0):
    #                 print("Showing only received frame numbers and supressing data descriptions")
    #             elif (print_level == 1):
    #                 print("Showing every frame")
    #             elif (print_level_mod == 1):
    #                 print("Showing every %dst frame"%print_level)
    #             elif (print_level_mod == 2):
    #                 print("Showing every %dnd frame"%print_level)
    #             elif (print_level == 3):
    #                 print("Showing every %drd frame"%print_level)
    #             else:
    #                 print("Showing every %dth frame"%print_level)

    #         elif c1 == 'q':
    #             is_looping = False
    #             streaming_client.shutdown()
    #             break
    #         else:
    #             print("Error: Command %s not recognized"%c1)
    #         print("Ready...\n")
    # print("exiting")
