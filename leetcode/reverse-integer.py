class Solution:
    def reverse(self, x: int) -> int:
        sign = -1 if '-' == str(x)[0] else 1
        if x:
            y = str(abs(x))[::-1]
            y = int(y.lstrip('0')) * sign
            if -pow(2, 31) < y < pow(2, 31) - 1:
                return y
            else:
                return 0
        else:
            return x * sign


if __name__ == '__main__':
    s = Solution()
    r = s.reverse(-0)
    print(r)
