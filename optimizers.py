from Memory import *
from torch.autograd import Variable
import torch

def optimize_model(model, optimizer, memory, batch_size=5, gamma=.99):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))


    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = torch.ByteTensor(tuple(map(lambda s: s[-1].value is not None, batch.future_state))).cuda()
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s[-1].value for s in batch.future_state
                                                if s[-1].value is not None]),
                                     volatile=True).cuda()
    state_batch = Variable(torch.cat([s.value for s in batch.state])).cuda()
    action_batch = Variable(torch.cat(batch.action)).cuda()
    reward_batch = Variable(torch.cat(batch.reward)).cuda()
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    action_res, autoencode_res = model(state_batch)
    state_action_values = action_res.gather(1, action_batch.view(-1,1)).cuda()

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size)).cuda()
    model_res, next_enc = model(non_final_next_states)
    next_state_values = model_res.max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss_auto = torch.sum(torch.pow(state_batch - autoencode_res, 2))
    print("encode_loss = ", loss_auto.data[0])
    print("loss = ", loss.data[0])
    # Optimize the model
    optimizer.zero_grad()
    loss = loss + loss_auto
    loss.backward()
    optimizer.step()
    #optimizer.zero_grad()
    # loss.backward()
    #optimizer.step()
