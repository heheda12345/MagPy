import torch
from frontend.compile import compile
from frontend.utils import SetConfig


class Example(torch.nn.Module):

    def __init__(self):
        super(Example, self).__init__()
        self.conv = torch.nn.Conv2d(3, 3, 3)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


with torch.no_grad():
    model = Example().eval()
    x = torch.randn(1, 3, 4, 4)
    expect_output = model(x)
    print("expect:", expect_output)

    # set the graph compiler to inductor
    with SetConfig({'backend': 'inductor'}):
        compiled = compile(model)
        # run the python code to compile the model. The fx graph and the guards will be printed out
        output1 = compiled(x)
        print("output1:", output1)

        # run the compiled model. "guard cache hit" means we find the compiled record and use it directly
        output2 = compiled(x)
        print("output2", output2)
        assert torch.allclose(expect_output, output1)
        assert torch.allclose(expect_output, output2)
