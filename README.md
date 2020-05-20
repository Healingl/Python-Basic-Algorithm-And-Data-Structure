#     Python-Basic-Algorithm-And-Data-Structure

python实现的基础数据结构和算法；

数据结构将涉及**顺序表、链表、堆栈、队列、树、二叉树、平衡二叉树、红黑树**；

算法将涉及**排序算法（冒泡排序、选择排序、插入排序、快速排序、希尔排序、归并排序）、查找算法（顺序查找、二分法查找、二叉树查找、哈希查找）**。



## 数据结构

### 顺序表

---





### 链表

---





### 堆栈

---





### 队列

---







### 树

---

树；是一种重要的非线性数据结构，直观地看，它是数据元素（在树中称为结点）按分支关系组织起来的结构，很象自然界中的树那样。

**树的一些基础概念：**

- 节点的度：一个节点含有的子树的个数称为该节点的度；
- 树的度：一棵树中，最大的节点的度称为树的度；
- 叶节点或终端节点：度为零的节点；
- 节点的层次：从根开始定义起，根为第1层，根的子节点为第2层，以此类推；
- 树的高度或深度：树中节点的最大层次；
- 森林：由m（m>=0）棵互不相交的树的集合称为森林；
- 路径：对于一棵子树中的任意两个不同的结点，如果从一个结点出发，按层次自上而下沿着一个个树枝能到达另一结点，称它们之间存在着一条路径

**常用树的分类：**

- 无序树：树中任意节点的子节点之间没有顺序关系，这种树称为无序树，也称为自由树；
- 有序树：树中任意节点的子节点之间有顺序关系，这种树称为有序树；
- 二叉树：每个节点最多含有两个子树的树称为二叉树；
- 完全二叉树：对于一颗二叉树，假设其深度为d(d>1)。除了第d层外，其它各层的节点数目均已达最大值，且第d层所有节点从左向右连续地紧密排列，这样的二叉树被称为完全二叉树，其中满二叉树的定义是所有叶节点都在最底层的完全二叉树;
- 平衡二叉树（AVL树）：当且仅当任何节点的两棵子树的高度差不大于1的二叉树；
- 排序二叉树（二叉查找树（英语：Binary Search Tree），也称二叉搜索树、有序二叉树）；
- 霍夫曼树（用于信息编码）：带权路径最短的二叉树称为哈夫曼树或最优二叉树；
- B树：一种对读写操作进行优化的自平衡的二叉查找树，能够保持数据有序，拥有多余两个子树。





**树的储存：**

树在python中可以通过列表和链表来储存，通过列表是将每个节点对象储存，在逻辑上不过形象，基本不用；用的最多的是通过链表构建一个树对象，其基本属性是根节点，根节点的左树属性和右树属性连接不同的节点，依次构建一颗庞大的树。



树的基本结构：

```
class Node(object):
    """节点类"""
    def __init__(self, elem=-1, lchild=None, rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild


class Tree(object):
    """树类"""
    def __init__(self, root=None):
        self.root = root
```



### 二叉树

---



二叉树的遍历：

二叉树是一种非常重要的数据结构，很多其它数据结构都是基于二叉树的基础演变而来的。对于二叉树，有深度遍历和广度遍历，**深度遍历有前序、中序以及后序三种遍历方法**，**广度遍历即我们平常所说的层次遍历**。

- 深度遍历

  ```
  class Node:
      """节点类"""
      def __init__(self, elem, lchild=None, rchild=None):
          self.elem = elem
          self.lchild = lchild
          self.rchild = rchild
   
  class Tree:
      """树类"""
      def __init__(self, root=None):
          self._root = root
   
      def add(self, item):
          node = Node(item)
          if not self._root:
              self._root = node
              return
          queue = [self._root]
          while queue:
              cur = queue.pop(0)
              if not cur.lchild:
                  cur.lchild = node
                  return
              elif not cur.rchild:
                  cur.rchild = node
                  return
              else:
                  queue.append(cur.rchild)
                  queue.append(cur.lchild)
   
      def preorder(self, root):
          """
          先序遍历-递归实现
          :param root:
          :return:
          """
          if root is None:
              return
          print(root.elem)
          self.preorder(root.lchild)
          self.preorder(root.rchild)
  			
   
      def inorder(self, root):
          """
          中序遍历-递归实现
          :param root:
          :return:
          """
          if root is None:
              return
          self.inorder(root.lchild)
          print(root.elem)
          self.inorder(root.rchild)
   
      def postorder(self, root):
          """
          后序遍历-递归实现
          :param root: 
          :return: 
          """
          if root is None:
              return
          self.postorder(root.lchild)
          self.postorder(root.rchild)
          print(root.elem)
          
       # 先序打印二叉树（非递归）
       #首先是先序遍历，需要借助一个堆栈，按照父亲节点、左孩子、右孩子的顺序压到堆里面，每次弹出栈顶元素 
  	def preorder(root): # 先序
          stack = []
          while stack or root:
              while root:
                  print(root.val)
                  stack.append(root)
                  root = root.lchild
              root = stack.pop()
              root = root.rchild
  
              
      # 中序打印二叉树（非递归）
      # 与先序遍历只是遍历的位置改变
      def inorder(root): # 中序
          stack = []
          while stack or root:
              while root:
                  stack.append(root)
                  root = root.lchild
              root = stack.pop()
              print(root.val)
              root = root.rchild
  
                  
      # 后序打印二叉树（非递归）
      # 最后到了后序遍历，这个有一点麻烦，需要好好理解，有左孩子就遍历左孩子，没有就转到右孩子
      def postorder(root): # 后序
          stack = []
          while stack or root:
              while root:
                  stack.append(root)
                  root = root.lchlid if root.lchild else root.right
              root = stack.pop()
              print(root.val)
              if stack and stack[-1].lchild == root:
                  root = stack[-1].rchild
              else:
                  root = None
  
                  
  ```
  
  
  
  
  
- 广度遍历

```
class Node:
    """节点类"""
    def __init__(self, elem, lchild=None, rchild=None):
        self.elem = elem
        self.lchild = lchild
        self.rchild = rchild
 
class Tree:
    """树类"""
    def __init__(self, root=None):
        self._root = root
 
    def breadth_travel(self, root):
        """
        广度优先-队列实现
        :param root:
        :return:
        """
        if not root:
            return 
        queue = [root]
        while queue:
            node = queue.pop(0)
            print(node.elem)
            if node.lchild:
                queue.append(node.lchild)
            elif node.rchild:
                queue.append(node.rchild)

```



二叉树常规操作：

```
# 统计树中节点个数
def count_BinTNodes(t):
    if t is None:
        return 0
    else:
        return 1 + count_BinTNode(t.left) \
               + count_BinTNode(t.right)
# 求二叉树所有数值和
def sum_BinTNodes(t):
    if t is None:
        return 0
    else:
        return t.dat + sum_BinTNodes(t.left) \
               + sum_BinTNodes(t.right
```

### 二叉搜索树（BST)

中序遍历团灭系列二叉搜索树问题

https://leetcode-cn.com/problems/minimum-absolute-difference-in-bst/solution/zhong-xu-bian-li-tuan-mie-xi-lie-er-cha-sou-suo-sh/

通过中序遍历二叉搜索树得到的关键码序列是一个递增序列。
这是二叉搜索树的一个重要性质，巧妙利用这一性质可以解决一系列二叉搜索树问题。
本系列以以下非递归中序遍历代码为核心，解决一系列相关问题。

```
p = root
st = []  # 用列表模拟实现栈的功能
while p is not None or st:
    while p is not None:
        st.append(p)
        p = p.left
    p = st.pop()
    proc(p.val)
    p = p.right
```

**一 二叉搜索树迭代器**
（一）算法思路
中序遍历二叉树
（二）算法实现

```
class BSTIterator:

    def __init__(self, root: TreeNode):
        self.root = root
        self.st = []
        self.current = self.root
        

    def next(self) -> int:
        """
        @return the next smallest number
        """
        while self.current is not None or self.st:
            while self.current is not None:
                self.st.append(self.current)
                self.current = self.current.left
            self.current = self.st.pop()
            node = self.current
            self.current = self.current.right
            return node.val
            
            

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        return self.current or self.st
```

**二 二叉搜索树的最小绝对差**
（一）算法思路
中序遍历二叉搜索树，第一个结点外的每个节点与其前一节点的差值，如果该值小于最小绝对差，就用它更新最小绝对差（初始可设为无穷）。
（二）算法实现

```
class Solution:
    def getMinimumDifference(self, root: TreeNode) -> int:
        st = []
        p = root
        pre = -float('inf')
        min_val = float('inf')
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            cur = p.val
            if cur - pre < min_val:
                min_val = cur - pre
            pre = cur
            p = p.right
        return min_val


```

（三）复杂度分析
时间复杂度：O(N)，N为树中节点个数。
空间复杂度：O(log(N))。



**三 二叉搜索树中第k小的元素**

（一）算法思路
二叉搜索树的中序遍历序列为递增序列，因此可中序遍历二叉搜索树，返回第K个元素。
（二）算法实现

```
class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        st = []
        p = root
        s = 0
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            s += 1
            if s == k:
                return p.val
            p = p.right

```

（三） 复杂度分析
时间复杂度：O(N)，N为树中节点个数。
空间复杂度：O(log(N))。



**四 二叉搜索树中的众数**

（一） 算法思想
二叉搜索树的中序遍历序列单调不减（或曰非严格单调递增），因此可考虑中序遍历二叉搜索树。
用max_times记录已访问节点的最大重复元素个数，time表示当前访问节点的元素值的出现次数,用res=[]记录结果。
若time == max_times，则将当前节点值添加到结果集。
若time > max_times，则以当前节点值构造新的列表作为结果，并用time更新max_times。
中序遍历结束后，返回结果res。

（二） 算法实现

```
class Solution:
    def findMode(self, root: TreeNode) -> List[int]:
        if root is None:
            return []
        p = root
        st = []
        res = []
        max_times = 1
        time = 1
        pre = float("inf")
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            
            cur = p.val
            if cur == pre:
                time += 1
            else:
                time = 1
                pre = cur
            if time == max_times:
                res.append(cur)
            if time > max_times:
                res = [cur]
                max_times = time
    
            p = p.right
                
        return res

```

（三） 复杂度分析
时间复杂度：O(N)，N为树中节点个数。
空间复杂度：最坏情况下为O(N)， 例如树畸形（树的高度为线性）或每个元素出现一次的情形。



**五 二叉搜索树的范围和**
（一）算法思路
中序遍历二叉搜索树
当节点的值等于L时开始累加，当节点的值等于R时停止累加并返回累加的结果。
（二）算法实现

```
class Solution:
    def rangeSumBST(self, root: TreeNode, L: int, R: int) -> int:
        st = []
        p = root
        s = 0
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            if p.val == L:
                s = L
                p = p.right
                break
            p = p.right
        
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            s += p.val
            if p.val == R:
                return s
            p = p.right
```

（三）复杂度分析
时间复杂度：O(N), N为树中节点数。
空间复杂度：O(log(N))。

**六 两数之和IV-输入BST**
（一）算法思路
中序遍历+双指针
（二）算法实现

```
class Solution(object):
    def findTarget(self, root, k):
        """
        :type root: TreeNode
        :type k: int
        :rtype: bool
        """
        nums = []
        st = []
        p = root
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            nums.append(p.val)
            p = p.right
        
        n = len(nums)
        i, j = 0, n-1
        while i < j:
            if nums[i] + nums[j] == k:
                return True
            elif nums[i] + nums[j] > k:
                j -= 1
            else:
                i += 1
        return False
```

（三）复杂度分析
时间复杂度：O(N)
空间复杂度：O(N)



**七 验证二叉搜索树**

（一）算法思路
一棵二叉树是二叉搜索树的充要条件是它的中序遍历序列单调递增，因此可以通过判断一个树的中序遍历序列是否单调递增来验证该树是否为二叉搜索树。
（二）算法实现

```
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def isValidBST(self, root):
        """
        :type root: TreeNode
        :rtype: bool
        """
        pre = -float('inf')
        p = root
        st = []
        while p is not None or st:
            while p is not None:
                st.append(p)
                p = p.left
            p = st.pop()
            if p.val > pre:
                pre = p.val
            else:
                return False
            p = p.right
        return True

```

（三）复杂度分析
时间复杂度：O(N)。
空间复杂度：O(log(N))。





### 平衡二叉树

---







### 红黑树

---









## 基础算法

### **深度优先搜索算法**

定义：一种用于遍历或搜索树或图的算法。 沿着树的深度遍历树的节点，尽可能深的搜索树的分支。当节点v的所在边都己被探寻过或者在搜寻时结点不满足条件，搜索将回溯到发现节点v的那条边的起始节点。整个进程反复进行直到所有节点都被访问为止。属于盲目搜索,最糟糕的情况算法时间复杂度为O(!n)。
