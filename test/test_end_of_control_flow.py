from frontend.bytecode_writter import get_instructions
from frontend.bytecode_analysis import end_of_control_flow


def while_loop():
    i = 0
    while i < 10:
        i += 1
        print(i)


def for_loop():
    for i in range(10):
        print(i)
    for j in (1, 2, 3):
        print(j)


def if_else():
    a = 1
    if a > 0:
        print("a > 0")
        if a > 1:
            print("a > 1")
    else:
        print("a <= 0")
        if a > -1:
            print("a > -1")
        else:
            print("a <= -1")


def forward_lstm(self, inputs):  # seq_len, batch, input_size
    state_c = ()
    state_h = ()
    for i in range(inputs.size()[0]):
        cur_input = inputs[i]
        for j in range(self.num_layers):
            c = cur_input
            h = c + cur_input
    return state_h[self.num_layers - 1]


def forward_seq2seq(self, encoder_output, std, h, c):
    batch_size = encoder_output.size()[1]
    cond = True
    id = 0
    while cond:
        x = self.embedding(output)
        id = id + 1
        cond = (torch.max(output) > self.EOS_token) & (id < self.max_length)
    return x


def forward_blockdrop(self, x, policy):

    x = self.seed(x)

    t = 0
    for segment, num_blocks in enumerate(self.layer_config):
        for b in range(num_blocks):
            action = policy[:, t].contiguous()
            residual = self.ds[segment](x) if b == 0 else x

            # early termination if all actions in the batch are zero
            if action.data.sum() == 0:
                x = residual
                t += 1
                continue

            action_mask = action.float().view(-1, 1, 1, 1)
            fx = F.relu(residual + self.blocks[segment][b](x))
            x = fx * action_mask + residual * (1 - action_mask)
            t += 1

    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    x = self.fc(x)
    return x


def check_one(f, start_pc: int, end_pc: int):
    instructions = get_instructions(f)
    end_pc_out = end_of_control_flow(instructions, start_pc)
    assert end_pc == end_pc_out, f"end_pc_ref: {end_pc}, end_pc_out: {end_pc_out}"


def test_end_of_control_flow():
    check_one(while_loop, 5, 15)
    check_one(for_loop, 4, 11)
    check_one(for_loop, 13, 20)
    check_one(if_else, 5, 36)
    check_one(if_else, 13, 36)
    check_one(if_else, 26, 36)
    check_one(forward_lstm, 12, 33)
    check_one(forward_lstm, 23, 32)
    check_one(forward_seq2seq, 11, 35)
    check_one(forward_blockdrop, 12, 99)
    check_one(forward_blockdrop, 20, 98)
    check_one(forward_blockdrop, 35, 44)
    check_one(forward_blockdrop, 51, 20)