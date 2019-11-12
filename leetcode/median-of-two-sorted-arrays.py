class Solution:
    def findMedianSortedArrays(self, nums1, nums2) -> float:
        len1 = len(nums1)
        len2 = len(nums2)
        length = len1 + len2
        if not length:
            return 0
        if length % 2:
            median = self.oddSolve(nums1, nums2, length)
        else:
            median = self.evenSolve(nums1, nums2, length)
        return float(median)

    def setParams(self, flag, i, nums, act):
        if act:
            if i + 1 == len(nums):
                flag = False
            else:
                i += 1
        return flag, i

    def oddSolve(self, nums1, nums2, length):
        stop = int((length + 1) / 2)
        if not nums1:
            return nums2[stop - 1]
        if not nums2:
            return nums1[stop - 1]
        eles = list()
        flag1 = True
        flag2 = True
        i1 = 0
        i2 = 0
        while len(eles) != stop:
            act1 = False
            act2 = False
            if flag1 and flag2:
                num1 = nums1[i1]
                num2 = nums2[i2]
                if num1 == num2:
                    if len(nums1) > len(nums2):
                        last = num1
                        act1 = True
                    elif len(nums1) < len(nums2):
                        last = num2
                        act2 = True
                elif num1 > num2:
                    last = num2
                    act2 = True
                elif num1 < num2:
                    last = num1
                    act1 = True
            elif flag1:
                last = nums1[i1]
                act1 = True
            elif flag2:
                last = nums2[i2]
                act2 = True
            eles.append(last)
            flag1, i1 = self.setParams(flag1, i1, nums1, act1)
            flag2, i2 = self.setParams(flag2, i2, nums2, act2)
        return eles[-1]

    def evenSolve(self, nums1, nums2, length):
        right = int(length / 2)
        left = right - 1
        stop = right + 1
        if not nums1:
            return (nums2[left] + nums2[right]) / 2.0
        if not nums2:
            return (nums1[left] + nums1[right]) / 2.0
        eles = list()
        flag1 = True
        flag2 = True
        i1 = 0
        i2 = 0
        while len(eles) != stop:
            act1 = False
            act2 = False
            if flag1 and flag2:
                num1 = nums1[i1]
                num2 = nums2[i2]
                if num1 == num2:
                    if len(nums1) >= len(nums2):
                        last = num1
                        act1 = True
                    elif len(nums1) < len(nums2):
                        last = num2
                        act2 = True
                elif num1 > num2:
                    last = num2
                    act2 = True
                elif num1 < num2:
                    last = num1
                    act1 = True
            elif flag1:
                last = nums1[i1]
                act1 = True
            elif flag2:
                last = nums2[i2]
                act2 = True
            eles.append(last)
            flag1, i1 = self.setParams(flag1, i1, nums1, act1)
            flag2, i2 = self.setParams(flag2, i2, nums2, act2)
        return (eles[-2] + eles[-1]) / 2.0


if __name__ == '__main__':
    s = Solution()
    r = s.findMedianSortedArrays([], [2, 3])
    print(r)
