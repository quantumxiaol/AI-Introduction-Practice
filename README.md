# AI-Introduction-Practice
人工智能导论实验课

第一题（10 分） 
关于搜索方面：采用 Python3.x 开发语言，编写程序实现一个简单吃豆人（Pac-Man）小游戏，界面采用 tkinter 包。白色为食物，黄色为 pacman，蓝色为幽灵 ghost。Pacman 通过移动（上下左右，有墙壁阻挡不可移动）可以吃掉食物，pacman 遇到幽灵会被吃掉，幽灵不吃食物，幽灵移动（上下左右，有墙壁阻挡不可移动）随机。以下两个问题至少选择一个问题使用算法自动实现。

（1）使用深度优先、广度优先、一致代价和 A*四种搜索策略搜索目标；

（2）使用 min-max 策略，pac-man 吃掉所有食物而不被幽灵吃掉。


程序采用tkinter开发，地图大小为31*31。


深度优先（Depth-First Search）
扩展最深的那个节点。使用LIFO栈。

广度优先（Breadth-First Search）
扩展最浅的那个节点。使用FIFO队列。

一致代价（Uniform-Cost Search）
扩展代价最小的那个节点。使用具有优先级的队列（优先级：累计代价/cumulative cost）

A*（）

UCS按照路径代价排序，即后向代价g(n) - Backward；贪心按照目标接近排序，即前向代价h(n) - Forward；A算法按照两者之和排序，即：f(n) = g(n) + h(n)

可相容的启发式方法是使用A*算法核心关键