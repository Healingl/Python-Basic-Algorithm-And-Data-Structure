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
  	def preOrderTravese(self, node):
  		"""
          先序遍历-非递归实现
          :param root: 
          :return: 
          """
          stack = [node]
          while len(stack) > 0:
              print(node.val)
              if node.right is not None:
                  stack.append(node.right)
              if node.left is not None:
                  stack.append(node.left)
              node = stack.pop()
              
      # 中序打印二叉树（非递归）
      def inOrderTraverse(self, node):
          stack = []
          pos = node
          while pos is not None or len(stack) > 0:
              if pos is not None:
                  stack.append(pos)
                  pos = pos.left
              else:
                  pos = stack.pop()
                  print(pos.val)
                  pos = pos.right
                  
      # 后序打印二叉树（非递归）
      # 使用两个栈结构
      # 第一个栈进栈顺序：左节点->右节点->跟节点
      # 第一个栈弹出顺序： 跟节点->右节点->左节点(先序遍历栈弹出顺序：跟->左->右)
      # 第二个栈存储为第一个栈的每个弹出依次进栈
      # 最后第二个栈依次出栈
      def postOrderTraverse(self, node):
          stack = [node]
          stack2 = []
          while len(stack) > 0:
              node = stack.pop()
              stack2.append(node)
              if node.left is not None:
                  stack.append(node.left)
              if node.right is not None:
                  stack.append(node.right)
          while len(stack2) > 0:
              print(stack2.pop().val)
                  
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





### 平衡二叉树

---







### 红黑树

---







