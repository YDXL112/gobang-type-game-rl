## 设计目标
- 在根目录新增 environment.py，提供 BatchTracker 类，用于批量 REINFORCE 训练。
- 完全遵循现有游戏规则：8×8 棋盘、不可落子区域为 0.1、玩家 1/-1 轮流落子、四连即胜、平局判定（棋盘填满或达步数上限）。
- 统一动作索引约定：idx = x*8 + y；同时支持 (x,y) 输入以适配不同上层代码。
- 向上提供批量接口：初始化、步进、重置、胜负判断、奖励、非法落子掩码。

## 数据结构
- board: [B, 8, 8] float32，初始模板与 chess_board.py 一致（含 0.1 不可落子区）。
- player_to_go: [B] int8，取值 {1, -1}。
- done: [B] bool。
- winners: [B] int8，终局为 {1, -1, 0}；未终局为 0。
- step_count: [B] int32。
- max_steps: int，默认 40，可配置。

## 公共接口
- __init__(batch_size: int, device=None, max_steps: int=40): 构造并初始化全部批次的棋盘与状态。
- reset(indices: Optional[Iterable[int]]=None): 重置整个批次或部分索引；返回重置后状态。
- state() -> Tensor: 返回当前 [B, 8, 8] 棋盘张量（可根据 device 返回 torch/numpy）。
- turn() -> Tensor: 返回 [B] 当前轮到玩家。
- illegal_mask() -> BoolTensor: 返回 [B, 8, 8]，True 表示不可落子（已占或 0.1）。
- legal_mask() -> BoolTensor: 返回 [B, 8, 8]，True 表示可落子（==0）。
- step(actions) -> (next_state, rewards, done, info):
  - actions 支持形状 [B]（整型 0..63）或 [B, 2]（x,y）。
  - 对于每个样本：若未终局且动作合法，落子 board[x,y]=player_to_go；随后切换轮到方。
  - 非法动作：保持棋面不变，标记 info['illegal'][b]=True；reward 仍按是否终局返回（非法本身不加额外惩罚）。
  - 内部调用 judge() 更新 winners/done；按映射 win=1、lose=-1、draw=0 返回 rewards（仅在终局返回终局奖励，未终局为 0）。
  - 返回 info: {'winners': [B], 'illegal': [B] Bool}。
- judge(indices: Optional[Iterable[int]]=None) -> winners: 批量四方向检测（水平/垂直/两对角），忽略 0 与 0.1，仅对 ±1 检测四连；更新 winners 与 done。

## 终止与奖励规则
- 终止条件：
  - 任一玩家形成四连。
  - 棋盘填满（无合法位置）。
  - 达到 max_steps（可配置，默认 40）。
- 奖励：
  - 胜者 +1，负者 -1，平局 0；在 step 返回时仅对已终局样本给出对应奖励，其它样本奖励为 0。

## 实现细节
- 初始化棋盘模板复用 chess_board.py 的 0.1 布局，批量复制到 [B,8,8]。
- 动作索引：若输入为 [B]，用 divmod(idx, 8) 得到 (x,y)；若输入为 [B,2]，直接读取。
- 合法性判定：legal_mask = (board==0)；illegal_mask = ~legal_mask。
- 四连检测：向量化滑窗检查四方向：
  - 水平：对每行 j:j+4；垂直：对每列 i:i+4；主对角与反对角：以 (i+k, j+k)/(i+k, j-k) 取 4 长度片段；分别比较是否全 1 或全 -1。
- 性能：优先使用 torch 在 GPU 上进行批量计算；若 device=None 则用 numpy。

## 使用示例（接口保证）
- 构造：env = BatchTracker(batch_size=32, device='cuda')
- 取掩码：mask = env.illegal_mask()  # [B,8,8]
- 采样动作（示例）：actions = sample(pi, mask)  # 上层策略按 mask 屏蔽非法位
- 步进：state, rewards, done, info = env.step(actions)
- 重置：env.reset(done.nonzero())  # 仅重置已终局样本

## 兼容性与扩展
- 统一 idx=x*8+y，避免现有代码里的两套索引不一致问题；若后续需要兼容旧逻辑，可在外层适配。
- 奖励仅在终局返回，步进不引入额外 shaping，便于与标准 REINFORCE 对齐；如需 shaping，可在上层策略添加。
- 可添加 get_available_actions(b) 返回每个样本的合法索引列表，方便策略采样。

请确认以上设计与接口。如果通过，我将按照此规范创建 environment.py 并实现 BatchTracker。