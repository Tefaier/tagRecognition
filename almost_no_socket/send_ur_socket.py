import socket

robotIP = "172.17.0.1"
REALTIME_PORT = 30003

urscript_command = '''
def myProg():
    textmsg("start")

    q_target = get_inverse_kin(p[0.1, 0.7, 0.1, 0, 0, 0])
    socket_open("192.168.56.1", 30009)

    socket_send_string(q_target)
    socket_close()

    textmsg("end")
end
myProg()
'''

def send_urscript_command(command: str):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((robotIP, REALTIME_PORT))

        command = command + "\n"
        
        s.sendall(command.encode('utf-8'))

        s.close()

    except Exception as e:
        print(f"An error occurred: {e}")

send_urscript_command(urscript_command)