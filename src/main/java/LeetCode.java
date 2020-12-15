import java.util.*;

/**
 * @author hao.peng01@hand-china.com 2020/12/8 9:38
 */
public class LeetCode {
    public static void main(String[] args) {
        String str = "2020-11-18 07:23:33";
//        DateTimeFormatter df = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
//        LocalDateTime preTime = LocalDateTime.parse(str, df);
//        LocalDateTime nextTime = LocalDateTime.now();
//        long days = Math.round(Math.ceil((double) ChronoUnit.HOURS.between(preTime, nextTime) / 24));
//        long days2 = ChronoUnit.DAYS.between(preTime, nextTime);
//        long hours = ChronoUnit.HOURS.between(preTime, nextTime);
//        System.out.println(preTime);
//        System.out.println(nextTime);
//        System.out.println(hours);
//        System.out.println(days);
//        System.out.println(days2);

        int x = 1;
        List<String> list = new ArrayList<>();
        System.out.println(x);
    }

    public int[] searchRange(int[] nums, int target) {
        int leftIdx = binarySearch(nums, target, true);
        int rightIdx = binarySearch(nums, target, false);
        if (leftIdx <= rightIdx && rightIdx < nums.length && nums[leftIdx] == target && nums[rightIdx] == target) {
            return new int[]{leftIdx, rightIdx};
        }
        return new int[]{-1, -1};
    }

    public int binarySearch(int[] nums, int target, boolean b) {
        int left = 0, right = nums.length - 1, ans = nums.length;
        while (left <= right) {
            int mid = (left + right) / 2;
            if (nums[mid] > target || (b && nums[mid] >= target)) {
                right = mid - 1;
                ans = mid;
            } else {
                left = mid + 1;
            }

        }
        return ans;
    }

    public void sortColors(int[] nums) {
        int n = nums.length;
        if (n < 2) {
            return;
        }
        int p = 0, q = nums.length - 1;
        for (int i = 0; i < q; i++) {
            if (nums[i] == 0) {
                nums[i] = nums[p];
                nums[p] = 0;
                p++;
            }
            if (nums[i] == 2) {
                nums[i] = nums[q];
                nums[q] = 0;
                q--;
            }
        }
    }

//    public char sslowestKey(int[] releaseTimes, String keysPressed) {
//        int maxTime = releaseTimes[0];
//        int len = releaseTimes.length;
//        for (int i = 1; i < len; i++) {
//            int tem = releaseTimes[i] - releaseTimes[i - 1];
//            if (tem > maxTime) {
//                maxTime = tem;
//            } else if (tem == maxTime) {
//                if (keysPressed.charAt(i) > keysPressed)
//            }
//
//        }
//
//    }

    private int[] sum;

//    public NumArray(int[] nums) {
//        sum = new int[nums.length + 1];
//        for (int i = 0; i < nums.length; i++) {
//            sum[i + 1] = sum[i] + nums[i];
//        }
//    }

    public int sumRange(int i, int j) {
        return sum[j + 1] - sum[i];
    }

    public int maxProfit(int[] prices) {
        int min = Integer.MAX_VALUE;
        int max = 0;
        int len = prices.length;
        for (int i = 0; i < len; i++) {
            if (prices[i] < min) {
                min = prices[i];
            }
            int tem = prices[i] - min;
            max = tem > max ? tem : max;
        }
        return max;
    }

    public int maxSubArray(int[] nums) {
        int pre = 0, max = nums[0];
        for (int x : nums) {
            pre = Math.max(pre + x, x);
            max = Math.max(max, pre);
        }
        return max;
    }

    public int countPrimes(int n) {
        int count = 0;
        boolean[] signs = new boolean[n];
        for (int i = 2; i * i < n; i++) {
            if (!signs[i]) {
                for (int j = i * i; j < n; j += i) {
                    signs[j] = true;
                }
            }
        }
        for (int i = 2; i < n; i++) {
            if (!signs[i]) {
                count++;
            }
        }
        return count;
    }

    public boolean isValid(String s) {
        int n = s.length();
        if (n % 2 == 1) {
            return false;
        }
        Map<Character, Character> map = new HashMap<>();
        map.put(')', '(');
        map.put(']', '[');
        map.put('}', '{');
        Deque<Character> stack = new LinkedList<>();
        for (int i = 0; i < n; i++) {
            char c = s.charAt(i);
            if (map.containsKey(s.charAt(i))) {
                if (stack.isEmpty() || !stack.peek().equals(map.get(c))) {
                    return false;
                }
                stack.pop();
            } else {
                stack.push(c);
            }
        }
        return stack.isEmpty();
    }

    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            if (map.containsKey(target - nums[i])) {
                return new int[]{map.get(target - nums[i]), i};
            }
            map.put(nums[i], i);
        }
        return new int[0];
    }

    public class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode() {
        }

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode left, TreeNode right) {
            this.val = val;
            this.left = left;
            this.right = right;
        }
    }

    public boolean isSymmetric(TreeNode root) {
        if (root == null) {
            return true;
        }
        return dfs(root.left, root.right);
    }

    boolean dfs(TreeNode left, TreeNode right) {
        if (left == null && right == null) {
            return true;
        }
        if (left == null || right == null) {
            return false;
        }
        if (left.val != right.val) {
            return false;
        }
        return dfs(left.left, right.right) && dfs(left.right, right.left);
    }

    public boolean isSubsequence(String s, String t) {
        int m = s.length();
        int n = t.length();
        int i = 0, j = 0;
        while (i < m && j < n) {
            if (s.charAt(m) == t.charAt(j)) {
                i++;
            } else {
                j++;
            }
        }
        return j == n;
    }

    public void moveZeroes(int[] nums) {
        int n = nums.length;
        int cur = 0;
        for (int i = 0; i < n; i++) {
            if (nums[i] != 0) {
                nums[cur++] = nums[i];
            }
        }
        for (int i = cur; i < n; i++) {
            nums[i] = 0;
        }
    }

    public List<Integer> findDisappearedNumbers(int[] nums) {
        for (int i = 0; i < nums.length; i++) {
            int newIndex = Math.abs(nums[i]) - 1;
            //如果存在某个数字X，就吧对应索引X-1的那个位置对应的数据变成负的
            //如果已经是负的就不管了
            if (nums[newIndex] > 0) {
                nums[newIndex] *= -1;
            }
        }
        List<Integer> result = new LinkedList<>();
        //遍历一遍，如果数组数据不为负，代表是缺少的
        //ps：也可以0到len-1，不过会add i+1，一样的
        for (int i = 1; i <= nums.length; i++) {
            if (nums[i - 1] > 0) {
                result.add(i);
            }
        }
        return result;
    }

    public class ListNode {
        int val;
        ListNode next;

        ListNode(int x) {
            val = x;
            next = null;
        }

        ListNode() {
        }

        ListNode(int val, ListNode next) {
            this.val = val;
            this.next = next;
        }

        public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
            if (headA == null || headB == null) {
                return null;
            }
            ListNode pa = headA;
            ListNode pb = headB;
            while (pa != pb) {
                pa = (pa == null ? headB : pa.next);
                pb = (pb == null ? headA : pb.next);
            }
            //给自己解释下结束条件
            //1.如果有相交，那不用说，a+b+c=b+a+c,pa等于pb了，所以循环结束
            //2.如果不相交，那就a+b=b+a，两个链表都被分别遍历两次，
            // 然后pa就等于pb等于null了，循环也结束了
            return pa;
        }

        public int rob(int[] nums) {
            int len = nums.length;
            if (len == 0) {
                return 0;
            }
            int f1 = 0;
            int f2 = 0;
            for (int i = 0; i < len; i++) {
                int tem = Math.max(f1 + nums[i], f2);
                f1 = f2;
                f2 = tem;
            }
            return Math.max(f1, f2);
        }

        public List<Integer> splitIntoFibonacci(String S) {
            List<Integer> list = new ArrayList<Integer>();
            backtrack(list, S, S.length(), 0, 0, 0);
            return list;
        }

        public boolean backtrack(List<Integer> list, String S, int length, int index, int sum, int prev) {
            if (index == length) {
                return list.size() >= 3;
            }
            long currLong = 0;
            for (int i = index; i < length; i++) {
                if (i > index && S.charAt(index) == '0') {
                    break;
                }
                currLong = currLong * 10 + S.charAt(i) - '0';
                if (currLong > Integer.MAX_VALUE) {
                    break;
                }
                int curr = (int) currLong;
                if (list.size() >= 2) {
                    if (curr < sum) {
                        continue;
                    } else if (curr > sum) {
                        break;
                    }
                }
                list.add(curr);
                if (backtrack(list, S, length, i + 1, prev + curr, curr)) {
                    return true;
                } else {
                    list.remove(list.size() - 1);
                }
            }
            return false;
        }
    }

    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<Integer>();
        inorder(root, res);
        return res;
    }

    public void inorder(TreeNode root, List<Integer> res) {
        if (root == null) {
            return;
        }
        inorder(root.left, res);
        res.add(root.val);
        inorder(root.right, res);
    }

    public int maximumWealth(int[][] accounts) {
        int m = accounts.length;
        int n = accounts[0].length;
        int max = 0;

        for (int i = 0; i < m; i++) {
            int tem = 0;
            for (int j = 0; j < n; j++) {
                tem += accounts[i][j];
            }
            if (max < tem) {
                max = tem;
            }
        }
        return max;
    }

    List<String> res = new ArrayList<>();
    int[] watch = new int[]{1, 2, 4, 8, 1, 2, 4, 8, 16, 32};
    int[] onOff = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    public List<String> readBinaryWatch(int num) {
        if (num > 8) {
            return res;
        }
        dfs(num, 0, 0);
        return res;
    }

    void dfs(int num, int onSum, int start) {
        if (onSum == num) {
            addResult();
            return;
        }
        for (int i = start; i < watch.length; i++) {
            onOff[i] = 1;
            dfs(num, onSum + 1, i + 1);
            onOff[i] = 0;
        }
    }

    public void addResult() {
        int hour = 0;
        int minute = 0;
        for (int i = 0; i < onOff.length; i++) {
            if (onOff[i] == 1) {
                if (i < 4) {
                    hour += watch[i];
                } else {
                    minute += watch[i];
                }
            }
        }
        if (hour > 11 || minute > 59) {
            return;
        }
        String time = hour + ":" + ((minute < 10) ? "0" + minute : minute);
        res.add(time);
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        backTrack(0, nums, res, new ArrayList<Integer>());
        return res;
    }

    private void backTrack(int i, int[] nums, List<List<Integer>> res, ArrayList<Integer> tmp) {
        res.add(new ArrayList<>(tmp));
        for (int j = i; j < nums.length; j++) {
            tmp.add(nums[j]);
            backTrack(j + 1, nums, res, tmp);
            tmp.remove(tmp.size() - 1);
        }
    }

    public int uniquePaths2(int m, int n) {
        int[] dp = new int[n];
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[j] = 1;
                } else {
                    dp[j] = dp[j - 1] + dp[j];
                }
            }
        }
        return dp[n - 1];
    }

    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; j++) {
                if (i == 0 || j == 0) {
                    dp[i][j] = 1;
                } else {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
        }
        return dp[m - 1][n - 1];
    }

    public List<String> generateParenthesis(int n) {
        List<String> ans = new ArrayList<>();
        backTrack(ans, new StringBuilder(), 0, 0, n);
        return ans;
    }

    private void backTrack(List<String> ans, StringBuilder cur, int open, int close, int max) {
        if (cur.length() == max * 2) {
            ans.add(cur.toString());
            return;
        }
        if (open < max) {
            cur.append('(');
            backTrack(ans, cur, open + 1, close, max);
            cur.deleteCharAt(cur.length() - 1);
        }
        if (open > close) {
            cur.append(')');
            backTrack(ans, cur, open, close + 1, max);
            cur.deleteCharAt(cur.length() - 1);
        }
    }

    public boolean lemonadeChange2(int[] bills) {
        int five = 0, ten = 0;
        for (int bill : bills) {
            if (bill == 5) {
                five++;
            } else if (bill == 10) {
                if (five == 0) {
                    return false;
                }
                ten++;
                five--;
            } else {
                if (five > 0 && ten > 0) {
                    five--;
                    ten--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }
        return true;
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        int len = candidates.length;
        List<List<Integer>> ans = new ArrayList<>();
        if (len == 0) {
            return ans;
        }
        Arrays.sort(candidates);
        Deque<Integer> path = new ArrayDeque<>();
        dfs(candidates, 0, len, target, path, ans);
        return ans;
    }

    private void dfs(int[] candidates, int begin, int len, int target, Deque<Integer> path, List<List<Integer>> ans) {
        if (target == 0) {
            ans.add(new ArrayList<>(path));
            return;
        }
        for (int i = begin; i < len; i++) {
            if (target - candidates[i] < 0) {
                break;
            }
            path.addLast(candidates[i]);
            dfs(candidates, i, len, target - candidates[i], path, ans);
            path.removeLast();
        }
    }

    public int[] dailyTemperatures(int[] T) {
        int[] res = new int[T.length];
        Deque<Integer> stack = new LinkedList<Integer>();
        for (int i = 0; i < T.length; i++) {
            while (!stack.isEmpty() && T[i] > T[stack.peek()]) {
                res[stack.peek()] = i - stack.pop();
            }
            stack.push(i);
        }
        return res;
    }

    //超时，没法ac
    public int trap2(int[] height) {
        int sum = 0;
        int max = getMax(height);//获取最大高度
        boolean isStart;
        int temp;
        for (int i = 0; i < max; i++) {
            isStart = false;
            temp = 0;
            for (int j = 0; j < height.length; j++) {
                if (isStart && height[j] < i) {
                    temp++;
                }
                if (height[j] >= i) {
                    sum += temp;
                    temp = 0;
                    isStart = true;
                }
            }
        }
        return sum;
    }

    private int getMax(int[] height) {
        int max = 0;
        for (int i = 0; i < height.length; i++) {
            if (height[i] > max) {
                max = height[i];
            }
        }
        return max;
    }

    public int trap(int[] height) {
        int ans = 0;
        Stack<Integer> stack = new Stack<>();
        int len = height.length;
        for (int i = 0; i < len; i++) {
            while (!stack.isEmpty() && height[stack.peek()] < height[i]) {
                int cur = stack.pop();
                if (stack.isEmpty()) {
                    break;
                }
                int l = stack.peek();
                int r = i;
                int h = Math.min(height[r], height[l]) - height[cur];
                ans += (r - l - 1) * h;
            }
            stack.push(i);
        }
        return ans;
    }

    public boolean lemonadeChange(int[] bills) {
        int[] money = new int[3];
        for (int i = 0; i < bills.length; i++) {
            switch (bills[i]) {
                case 5:
                    money[0]++;
                    break;
                case 10:
                    money[1]++;
                    if (money[0] > 0) {
                        money[0]--;
                    } else {
                        return false;
                    }
                    break;
                case 20:
                    money[2]++;
                    if (money[1] > 0 && money[0] > 0) {
                        money[0]--;
                        money[1]--;
                    } else if (money[0] > 3) {
                        money[0] -= 3;

                    } else {
                        return false;
                    }
                    break;
            }
        }
        return true;
    }


    public int[] countBits2(int num) {
        int[] ans = new int[num + 1];
        for (int i = 0; i <= num; i++) {
            ans[i] = Integer.bitCount(i);
        }
        return ans;
    }

    public int[] countBits(int num) {
        int[] ans = new int[num + 1];
        int i = 0, b = 1;
        while (b <= num) {
            while (i < b && i + b <= num) {
                ans[i + b] = ans[i] + 1;
                i++;
            }
            i = 0;
            b <<= 1;
        }
        return ans;
    }

    public int[][] reconstructQueue(int[][] people) {
        Arrays.sort(people, new Comparator<int[]>() {
            @Override
            public int compare(int[] person1, int[] person2) {
                if (person1[0] != person2[0]) {
                    return person2[0] - person1[0];
                } else {
                    return person1[1] - person2[1];
                }
            }
        });
        System.out.println(people.toString());
        List<int[]> ans = new ArrayList<>();
        for (int[] person : people) {
            ans.add(person[1], person);
        }
        return ans.toArray(new int[ans.size()][]);
    }

    public int rob(TreeNode root) {
        int[] rootStatus = deep(root);
        return Math.max(rootStatus[0], rootStatus[1]);
    }

    private int[] deep(TreeNode root) {
        if (root == null) {
            return new int[]{0, 0};
        }
        int[] left = deep(root.left);
        int[] right = deep(root.right);

        int selected = root.val + left[1] + right[1];
        int notSelected = Math.max(left[0], left[1]) + Math.max(right[0], right[1]);
        return new int[]{selected, notSelected};
    }

    public String predictPartyVictory(String senate) {
        int n = senate.length();
        Queue<Integer> radiant = new LinkedList<Integer>();
        Queue<Integer> dire = new LinkedList<Integer>();
        for (int i = 0; i < n; i++) {
            if (senate.charAt(i) == 'R') {
                radiant.offer(i);
            } else {
                dire.offer(i);
            }
        }
        while (!radiant.isEmpty() && !dire.isEmpty()) {
            int radiantIndex = radiant.poll();
            int direIndex = dire.poll();
            if (radiantIndex < direIndex) {
                radiant.offer(radiantIndex + n);
            } else {
                dire.offer(direIndex + n);
            }
        }
        return radiant.isEmpty() ? "Dire" : "Radiant";
    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> output = new ArrayList<>();
        for (int num : nums) {
            output.add(num);
        }
        int n = nums.length;
        backTrack(n, output, res, 0);
        return res;
    }

    public void backTrack(int n, List<Integer> output, List<List<Integer>> res, int first) {
        if (first == n) {
            res.add(new ArrayList<>(output));
            return;
        }
        for (int i = first; i < n; i++) {
            Collections.swap(output, first, i);
            backTrack(n, output, res, first + 1);
            Collections.swap(output, first, i);
        }
    }

    private Stack<int[]> stack = new Stack<>();

    public void push(int x) {
        if (stack.isEmpty()) {
            stack.push(new int[]{x, x});
        } else {
            stack.push(new int[]{x, Math.min(x, stack.peek()[1])});
        }
    }

    public void pop() {
        stack.pop();
    }

    public int top() {
        return stack.peek()[0];
    }

    public int getMin() {
        return stack.peek()[1];
    }

    public String removeKdigits(String num, int k) {
        int len = num.length();
        Deque<Character> deque = new LinkedList<>();
        for (int i = 0; i < len; i++) {
            char digit = num.charAt(i);
            while (!deque.isEmpty() && k > 0 && deque.peekLast() > digit) {
                deque.pollLast();
                k--;
            }
            deque.offerLast(digit);
        }
        for (int i = 0; i < k; ++i) {
            deque.pollLast();
        }
        StringBuilder ret = new StringBuilder();
        boolean leadingZero = true;
        while (!deque.isEmpty()) {
            char digit = deque.pollFirst();
            if (leadingZero && digit == '0') {
                continue;
            }
            leadingZero = false;
            ret.append(digit);
        }
        return ret.length() == 0 ? "0" : ret.toString();
    }

    //        public boolean isPalindrome(ListNode head) {
//            List<Integer> vals = new ArrayList<>();
//
//        }
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }

    public int numTrees(int n) {
        int[] G = new int[n + 1];
        G[0] = 1;
        G[1] = 1;
        for (int i = 2; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                G[i] = G[j - 1] * G[i - j];
            }
        }
        return G[n];
    }

    public int lengthOfLIS(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        int maxLength = 1;
        int[] dp = new int[nums.length];
        dp[0] = 1;
        for (int i = 1; i < nums.length; i++) {
            dp[i] = 1;
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            maxLength = Math.max(maxLength, dp[i]);
        }
        return maxLength;
    }

    public int findKthLargest(int[] nums, int k) {
        Arrays.sort(nums);
        return nums[nums.length - k];
    }

    public void flatten(TreeNode root) {
        if (root == null) {
            return;
        }
        flatten(root.left);
        TreeNode temp = root.right;
        root.right = root.left;
        root.left = null;
        while (root.right != null) {
            root = root.right;
        }
        flatten(temp);
        root.right = temp;
    }

    public int monotoneIncreasingDigits(int N) {
        char[] str = Integer.toString(N).toCharArray();
        int i = 1;
        while (i < str.length && str[i - 1] <= str[i]) {
            i++;
        }
        if (i < str.length) {
            while (i > 0 && str[i - 1] > str[i]) {
                str[i - 1]--;
                i--;
            }
            for (i++; i < str.length; i++) {
                str[i] = '9';
            }
        }
        return Integer.parseInt(new String(str));
    }

    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode pre = new ListNode();
        ListNode cur = pre;
        int carry = 0;
        while (l1 != null || l2 != null) {
            int x = l1 == null ? 0 : l1.val;
            int y = l2 == null ? 0 : l2.val;
            int sum = x + y + carry;

            carry = sum > 9 ? 1 : 0;
            sum = sum % 10;
            cur.next = new ListNode(sum);
            cur = cur.next;
            if (l1 != null) {
                l1 = l1.next;
            }
            if (l2 != null) {
                l2 = l2.next;
            }
        }
        if (carry == 1) {
            cur.next = new ListNode(carry);
        }
        return pre.next;
    }


    public int lengthOfLongestSubstring(String s) {
        Set<Character> occ = new HashSet<>();
        int n = s.length();
        int rk = -1, ans = 0;
        for (int i = 0; i < n; i++) {
            if (i != 0) {
                occ.remove(s.charAt(i - 1));
            }
            while (rk + 1 < n && !occ.contains(s.charAt(rk + 1))) {
                occ.add(s.charAt(rk + 1));
                rk++;
            }
            ans = Math.max(ans, rk - i + 1);
        }
        return ans;
    }

    public int minCostClimbingStairs2(int[] cost) {
        int len = cost.length;
        int f1 = 0, f2 = 0;
        for (int i = 0; i < len; i++) {
            int f0 = cost[i] + Math.min(f1, f2);
            f1 = f2;
            f2 = f0;
        }
        return Math.min(f1, f2);
    }

    public int minCostClimbingStairs(int[] cost) {
        int len = cost.length;
        int f1 = 0, f2 = 0;
        for (int i = len - 1; i >= 0; i--) {
            int f0 = cost[i] + Math.min(f1, f2);
            f2 = f1;
            f1 = f0;
        }
        return Math.min(f1, f2);
    }

    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> ans = new ArrayList<>();
        for (int i = 0; i < numRows; i++) {
            List<Integer> row = new ArrayList<>();
            for (int j = 0; j <= i; j++) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(ans.get(i - 1).get(j - 1) + ans.get(i - 1).get(j));
                }
            }
            ans.add(row);
        }
        return ans;
    }

    public int leastInterval(char[] tasks, int n) {
        Map<Character, Integer> freq = new HashMap<Character, Integer>();
        // 最多的执行次数
        int maxExec = 0;
        for (char ch : tasks) {
            int exec = freq.getOrDefault(ch, 0) + 1;
            freq.put(ch, exec);
            maxExec = Math.max(maxExec, exec);
        }

        // 具有最多执行次数的任务数量
        int maxCount = 0;
        Set<Map.Entry<Character, Integer>> entrySet = freq.entrySet();
        for (Map.Entry<Character, Integer> entry : entrySet) {
            int value = entry.getValue();
            if (value == maxExec) {
                ++maxCount;
            }
        }

        return Math.max((maxExec - 1) * (n + 1) + maxCount, tasks.length);
    }


    //详细
    public int matrixScore2(int[][] A) {
        int m = A.length, n = A[0].length;
        //step1:翻转所有行，保证第一列全为1
        for (int i = 0; i < m; i++) {
            if (A[i][0] != 1) {
                //翻转该行
                for (int j = 0; j < n; j++) {
                    A[i][j] = 1 - A[i][j];
                }
            }
        }
        //step2:从第二列开始，计算每一列1的个数count，若count<=A行数的一半，则说明需要进行翻转
        for (int j = 1; j < n; j++) {
            int count = 0;
            for (int i = 0; i < m; i++) {
                if (A[i][j] == 1) {
                    count++;
                }
            }
            if (count < (m + 1) / 2) {
                for (int i = 0; i < m; i++) {
                    A[i][j] = 1 - A[i][j];
                }
            }
        }
        //step3:计算结果并返回
        int num = 0;
        for (int j = 0; j < n; j++) {
            int temp = (int) Math.pow(2, n - j - 1);
            for (int i = 0; i < m; i++) {
                num += A[i][j] * temp;
            }
        }
        return num;
    }

    public int matrixScore(int[][] A) {
        int m = A.length, n = A[0].length;
        for (int[] a : A) {
            if (a[0] == 0) {
                for (int i = 0; i < n; i++) {
                    a[i] = 1 - a[i];
                }
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            int count = 0;
            for (int j = 0; j < m; j++) {
                if (A[j][i] == 1) {
                    count++;
                }
            }
            if (count <= m / 2) {
                count = m - count;
            }
            ans += count * Math.pow(2, n - i - 1);
        }
        return ans;
    }

    public boolean isPossible(int[] nums) {
        int n = nums.length;
        int dp1 = 0;    // 长度为1的子序列数目
        int dp2 = 0;    // 长度为2的子序列数目
        int dp3 = 0;    // 长度>=3的子序列数目
        int idx = 0;
        int start = 0;  // 标记子序列的起始位置

        while (idx < n) {
            start = idx;                        // 重新将起始位置赋值
            int x = nums[idx];
            while (idx < n && nums[idx] == x) {   // 去掉所有和x重复的元素
                idx++;
            }
            int cnt = idx - start;

            if (start > 0 && x != nums[start - 1] + 1) {
                if (dp1 + dp2 > 0) {
                    return false;
                } else {
                    dp1 = cnt;
                    dp2 = dp3 = 0;
                }
            } else {
                if (dp1 + dp2 > cnt) {
                    return false;
                }
                int left = cnt - dp1 - dp2;
                int keep = Math.min(dp3, left);
                dp3 = keep + dp2;


                dp2 = dp1;
                dp1 = left - keep;
            }
        }

        return dp1 == 0 && dp2 == 0;
    }

    int ans = 0;

    public int diameterOfBinaryTree(TreeNode root) {
        depth(root);
        return ans;
    }

    public ListNode sortList(ListNode head) {
        if (head == null) {
            return head;
        }
        int length = 0;
        ListNode node = head;
        while (node != null) {
            length++;
            node = node.next;
        }
        ListNode dummyHead = new ListNode(0, head);
        for (int subLength = 1; subLength < length; subLength <<= 1) {
            ListNode prev = dummyHead, curr = dummyHead.next;
            while (curr != null) {
                ListNode head1 = curr;
                for (int i = 1; i < subLength && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode head2 = curr.next;
                curr.next = null;
                curr = head2;
                for (int i = 1; i < subLength && curr != null && curr.next != null; i++) {
                    curr = curr.next;
                }
                ListNode next = null;
                if (curr != null) {
                    next = curr.next;
                    curr.next = null;
                }
                ListNode merged = merge(head1, head2);
                prev.next = merged;
                while (prev.next != null) {
                    prev = prev.next;
                }
                curr = next;
            }
        }
        return dummyHead.next;
    }

    public ListNode merge(ListNode head1, ListNode head2) {
        ListNode dummyHead = new ListNode(0);
        ListNode temp = dummyHead, temp1 = head1, temp2 = head2;
        while (temp1 != null && temp2 != null) {
            if (temp1.val <= temp2.val) {
                temp.next = temp1;
                temp1 = temp1.next;
            } else {
                temp.next = temp2;
                temp2 = temp2.next;
            }
            temp = temp.next;
        }
        if (temp1 != null) {
            temp.next = temp1;
        } else if (temp2 != null) {
            temp.next = temp2;
        }
        return dummyHead.next;
    }


    public int depth(TreeNode node) {
        if (node == null) {
            return 0;
        }
        int left = depth(node.left);
        int right = depth(node.right);
        ans = Math.max(left + right, ans);
        return Math.max(left, right) + 1;
    }

    public boolean hasCycle2(ListNode head) {
        Set<ListNode> set = new HashSet<>();
        while (head != null) {
            if (!set.add(head)) {
                return true;
            }
        }
        return false;
    }

    public boolean hasCycle(ListNode head) {
        if (head == null || head.next == null) {
            return false;
        }
        ListNode slow = head;
        ListNode fast = head.next;
        while (slow != fast) {
            if (fast == null || fast.next == null) {
                return false;
            }
            fast = fast.next.next;
            slow = slow.next;
        }
        return true;
    }

    public boolean isPalindrome(ListNode head) {
        if (head == null) {
            return true;
        }
        ListNode firstHalfEnd = endOfFirstHalf(head);
        ListNode secondHalfStart = reverseList(firstHalfEnd.next);
        ListNode p1 = head;
        ListNode p2 = secondHalfStart;
        boolean result = true;
        while (result && p2 != null) {
            if (p1.val != p2.val) {
                result = false;
            }
            p1 = p1.next;
            p2 = p2.next;
        }
        // 还原链表并返回结果
        firstHalfEnd.next = reverseList(secondHalfStart);
        return result;
    }

    public ListNode reverseList(ListNode head) {
        ListNode prev = null;
        ListNode curr = head;
        while (curr != null) {
            ListNode tem = curr.next;
            curr.next = prev;
            prev = curr;
            curr = tem;
        }
        return prev;

    }

    public ListNode endOfFirstHalf(ListNode head) {
        ListNode fast = head;
        ListNode slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        return slow;
    }

    public boolean searchMatrix(int[][] matrix, int target) {
        int m = matrix.length;
        if (m == 0) {
            return false;
        }
        int n = matrix[0].length;
        int left = 0, right = m * n - 1;
        int pId, pEl;
        while (left <= right) {
            pId = (left + right) / 2;
            pEl = matrix[pId / n][pId % n];
            if (pEl == target) {
                return true;
            } else if (pEl < target) {
                left = pId + 1;
            } else {
                right = pId - 1;
            }
        }
        return false;
    }
}
