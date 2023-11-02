import dataclasses
from typing import Any, Optional
import torch
from .store_pos import StorePos


@dataclasses.dataclass
class LoopPosMap:
    input_only_pos: list[tuple[str, StorePos]]
    joint_pos: list[tuple[str, StorePos]]
    output_only_pos: list[tuple[str, StorePos]]


class LoopModule(torch.nn.Module):  #type: ignore
    body: torch.fx.GraphModule
    num_read_only_param: int
    num_iter: int

    def __init__(self, body: torch.fx.GraphModule, num_read_only_param: int,
                 num_iter: int):
        super(LoopModule, self).__init__()
        self.body = body
        self.num_read_only_param = num_read_only_param
        self.num_iter = num_iter

    # def forward(self, num_iter: Optional[int], cond: torch.Tensor, *values:
    #             Any) -> Any:
    def forward(self, *values: Any) -> Any:
        iter_num = 0
        # assert cond.dtype == torch.bool
        read_only = values[:self.num_read_only_param]
        loop_carry = values[self.num_read_only_param:]
        while iter_num < self.num_iter:
            # and cond.item():
            loop_carry = self.body(torch.tensor(iter_num), *read_only,
                                   *loop_carry)
            # cond, *loop_carry = self.body(iter_num, cond, *read_only,
            #                               *loop_carry)
            iter_num += 1
        return loop_carry


class ControlFlowInfo:
    start_pc: int
    end_pc: int

    def __init__(self, start_pc: int, end_pc: int) -> None:
        self.start_pc = start_pc
        self.end_pc = end_pc


class ForLoopInfo(ControlFlowInfo):
    num_iter: int
    cur_iter: int
    pos_map: Optional[LoopPosMap]
    inner_graph: Optional[torch.fx.Graph]

    def __init__(self, start_pc: int, end_pc: int, num_iter: int) -> None:
        super().__init__(start_pc, end_pc)
        self.num_iter = num_iter
        self.cur_iter = 0
