from numpy import random as rn
timestamps = 100
p_packet = 0.15
packet_length = 5

class TransmitterNode:
    def __init__(self, p_packet=0.3, channel=None, packet_length = 5, name="0"):
        self.p_packet = p_packet
        self.channel = channel
        self.packet_length = packet_length
        self.name = name
        
    def send_data(self, i):
        SEND_DATA = rn.choice([0,1], p=[1-p_packet, p_packet])
        print("Node:", self.name, "The channel at", i, "is:", channel[i])
        if not channel[i]:
            if SEND_DATA:
                print("Node", self.name, "decided to send data at timestamp", i)
                for j in range(i, i+self.packet_length):
                    try:
                        channel[j] = 1
                    except:
                        pass
    
channel = [0 for i in range(timestamps)]
print(channel)
Node1 = TransmitterNode(p_packet, channel, packet_length, "1")
Node2 = TransmitterNode(p_packet, channel, packet_length, "2")
for i in range(timestamps):
    Node1.send_data(i)
    Node2.send_data(i)
print(channel)