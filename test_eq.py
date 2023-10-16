import torch
from dendsn.model.dend_compartment import PassiveDendCompartment


print("Test 1: PassiveDendCompartment, v_rest=v_init=0")
T = 16
S = [4, 28, 28]
TAU = 3.

x_seq = torch.randn(size=[T, *S])
dc = PassiveDendCompartment(tau=TAU, decay_input=False, step_mode="s")

single_step_result = []
for t in range(T):
    single_step_result.append(dc(x_seq[t]))
single_step_result = torch.stack(single_step_result)
dc.reset()

dc.step_mode = "m"
dc.store_v_seq = True
multi_step_result = dc(x_seq)
multi_step_v_seq = dc.v_seq
last_step_v = dc.v

cond1 = torch.all(torch.abs(single_step_result- multi_step_result) < 1e-4)
cond2 = torch.all(torch.eq(multi_step_v_seq[-1], last_step_v))
cond3 = torch.all(torch.eq(multi_step_result, multi_step_v_seq))
print(cond1, cond2, cond3)


print("Test 2: PassiveDendCompartment, v_rest=v_init!=0")
T = 16
S = [1]
TAU = 2.

x_seq = torch.randn(size=[T, *S])
dc = PassiveDendCompartment(tau=TAU, v_rest=1.0, decay_input=False, step_mode="s")

single_step_result = []
for t in range(T):
    single_step_result.append(dc(x_seq[t]))
single_step_result = torch.stack(single_step_result)
dc.reset()

dc.step_mode = "m"
dc.store_v_seq = True
multi_step_result = dc(x_seq)
multi_step_v_seq = dc.v_seq
last_step_v = dc.v

cond1 = torch.all(torch.abs(single_step_result- multi_step_result) < 1e-4)
cond2 = torch.all(torch.eq(multi_step_v_seq[-1], last_step_v))
cond3 = torch.all(torch.eq(multi_step_result, multi_step_v_seq))
print(cond1, cond2, cond3)
