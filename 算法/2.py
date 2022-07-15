from typing import Optional


# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode], carry=0) -> Optional[ListNode]:
        # # 根据递归的解法进行，考虑进位，添加一个参数
        # # 获取当下的值
        # val1 = l1.val if l1 else 0
        # val2 = l2.val if l2 else 0
        # # 计算进位与第一位的值
        # val = val1 + val2 + carry
        # val, carry = val % 10, 1 if val > 9 else 0
        # # 获取next
        # next1, next2 = l1.next if l1 else None, l2.next if l2 else None
        # # 进入递归的条件判断
        # if next1 or next2 or carry:
        #     return ListNode(val, self.addTwoNumbers(next1, next2, carry))
        # return ListNode(val)

        # 实例化类，创建链表，链式赋值
        res = head = ListNode()
        carry = 0
        # 循环条件
        while l1 or l2 :
            # 获取当前位的值
            val1, val2 = l1.val if l1 else 0, l2.val if l2 else 0
            # 计算总值 + 进位
            val = val1 + val2 + carry
            carry = val // 10
            # 链表赋值
            res.next = ListNode(val % 10)
            if l1: l1 = l1.next
            if l2: l2 = l2.next
            res = res.next

        if carry: res.next = ListNode(carry)

        return head.next
