# 给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
#
# 你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。
#
# 你可以按任意顺序返回答案。
#
# 来源：力扣（LeetCode）
# 链接：https://leetcode.cn/problems/two-sum
# 著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。

# 入门第一题，这道题看到第一眼就知道二重遍历可以解决，但是也可以想到时间复杂O(n2)，太慢，所以切到另一个思路，思路核心是字典查询速度为O(1)

class Solution:
    def twoSum(self, nums, target):
        """
        整体复杂度为O(n*1)
        :param nums:
        :param target:
        :return:
        """
        temp = {}
        for index, num in enumerate(nums):
            key = target - num
            if key in temp:
                return [temp[key], index]
            else:
                temp[num] = index



if __name__ == '__main__':
    pass
