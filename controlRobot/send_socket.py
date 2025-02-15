import socket
import get_info_topic_joint_states
import array

robotIP = "127.0.0.1"
REALTIME_PORT = 30003

# q_target = get_inverse_kin(target_pose) 
urscript_command = '''
def myProg():
    target_pose = p[70, 0.1, 0.1, 0, 0, 0]
    success = is_within_safety_limits(target_pose)

    if success:
        movej(target_pose, a=1.2, v=0.25, r = 0)
        textmsg("ok1")
    else:
        textmsg("bad")
    end
end
myProg()
'''


def send_urscript_command(command: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((robotIP, REALTIME_PORT))

        command = command + "\n"
        
        s.sendall(command.encode('utf-8'))

        # ans = s.recv(1220)

        s.close()

    except Exception as e:
        print(f"An error occurred: {e}")

send_urscript_command(urscript_command)

while True:
    js = get_info_topic_joint_states.get_joint_state(timeout=5.0)
    if js.velocity == array.array('d', [0, 0, 0, 0, 0, 0]):
        # with open("/home/mathew131/ur_programs/output.txt", "w") as file:
        #     file.write(str([*js.position]))
        print("Полученные положения суставов:", js.position)
        break