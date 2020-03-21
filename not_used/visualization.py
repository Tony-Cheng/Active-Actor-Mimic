import matplotlib.pyplot as plt
from IPython import display
import time


def visualize_AMN(AMN_network, memory, expert_network=None,device='cuda'):
    states, _, _, _, _ = memory.sample()
    images = []
    for i in range(50):
        time.sleep(0.5)
        AMN_prediction = AMN_network(states[i:i+1].to(device))
        if expert_network is not None:
          expert_prediction = expert_network(states[i:i+1].to(device))
        for j in range(4):
            time.sleep(0.1)
            plt.figure(3)
            plt.clf()
            images.append(states[i,j].numpy())
            plt.imshow(states[i,j].numpy())
            if expert_network is not None:
                plt.title("AMN: %s expert: %s" % (AMN_prediction, expert_prediction))
            else:
                plt.title("AMN: %s" % (AMN_prediction))
            plt.axis('off')
            
            display.clear_output(wait=True)
            display.display(plt.gcf())
    return images


