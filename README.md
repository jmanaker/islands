# islands
Solves a puzzle I saw [https://codereview.stackexchange.com/questions/191747/c-interview-problem-finding-an-island-in-a-2d-grid/ posted at codereview.stackexchange.com]:
> You are given a field of 0-1 values on a finite two-dimensional grid of size N by M, presumably selected randomly from the Bernoulli measure.  Count the number of "islands" of contiguous 1s, where two elements are adjacent iff they are to the left, right, up, or down, but not diagonal.

The given solution used a naive recursion, which is slow, slow, slow!  This is a much faster version that works line-by-line with a tree cache for merging islands, as well as the test code to verify that it is indeed faster.  I haven't tried to memoize the naive recursion, but I don't think it helps.
In the process, I also ended up effectively (re)inventing the C++ stdlib's std::valarray, about a year or two in advance.  Unless your compiler is really old, you should use that instead.  

(Currently has a history branch that main needs to merge atop.)  
