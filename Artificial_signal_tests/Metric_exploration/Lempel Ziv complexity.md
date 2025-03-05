Lempel Ziv complexity is a timeseries measure originally designed for binary sequences. It can be applied to any timeseries signal by symbolizing the signal, mostly arround its mean. It is commonly used within timeseries analysis as a measure for the complexity of a series of symbols. It's basic working mechanism lies in finding a unique set of substrings that the full signal is comprised off. Another very common mode of application lies in the compression algorithm LZ76 which uses this method to compress timeseries using these unique individual substrings


I find the mathematics behind it still quite hard to grasp sometimes as I struggle a bit with the pattern matching part

TODO: take a deeper look again at the algorithm and how it finds the substrings / which substrings it extracts
-> with this also look at how the compression of the string works and how it relates to the final complexity of the string

I went ahead and analyzed the version from neurokit2. The code was allready well documented

chat gpt explanation from proper search string:
### **Step 1: Initializing the Process**

We start by comparing the first bits:
Here’s an improved version incorporating all elements of your notation while keeping the explanation clear and structured:

---

### **Applying the Lempel-Ziv Algorithm to a Signal**

We begin with the binary signal:

```
0001100110
```

The Lempel-Ziv compression method identifies repeated patterns in the signal and represents them efficiently. The comparison process is visualized in three rows:

1. **Look-Ahead (LA) Buffer:** Denoted by `-` and `v`, where `v` represents the length of the currently matched pattern.
2. **Binary Signal:** The original sequence being processed.
3. **Pointer Position:** Denoted by `^`, where # of `^` symbols indicate the current matched pattern length.

---

### **Step 1: Initializing the Process**

We start by comparing the first bits:

```
-v
0001100110
^
```

- Matching segment: `[(1, '0'), (2, '0')]`
- The bits match, increasing the matched pattern length.

---

### **Step 2: Expanding the Match**

We continue extending the match:

```
-vv
0001100110
^^
```

- Matching segment: `[(2, '0'), (3, '0')]`
- The bits match, increasing the pattern length.

```
-vvv
0001100110
^^^
```

- Matching segment: `[(3, '0'), (4, '1')]`
- The bits **do not** match.

Since we found a pattern (`000`) of length **3**, we:

- Update **max pattern length** to 3.
- Move the pointer forward in the **Search Buffer (SB)**.
- Reach the **Look-Ahead (LA) region**, meaning we store `000` and reset the search.

---

### **Step 3: Searching for a New Pattern**

Restarting the search:

```
----v
0001100110
^
```

- Comparing `[(1, '0'), (5, '1')]`, the bits **do not** match.
- Move the pointer forward.

```
----v
0001100110
-^
```

- Comparing `[(2, '0'), (5, '1')]`, no match.

```
----v
0001100110
--^
```

- Comparing `[(3, '0'), (5, '1')]`, no match.

```
----v
0001100110
---^
```

- Comparing `[(4, '1'), (5, '1')]`, **match found**.

```
----vv
0001100110
---^^
```

- Comparing `[(5, '1'), (6, '0')]`, **no match**.

Since the **max matching length** is **2**, we store this pattern (`11`), reset, and continue.

---

### **Step 4: Identifying the Final Pattern**

We now analyze the last segment:

```
------v
0001100110
^
```

- Comparing `[(1, '0'), (7, '0')]`, **match found**.

```
------vv
0001100110
^^
```

- Comparing `[(2, '0'), (8, '1')]`, **no match**.
- Store `00` and reset.

Restarting:

```
------v
0001100110
-^
```

- Comparing `[(2, '0'), (7, '0')]`, **match found**.

```
------vv
0001100110
-^^
```

- Comparing `[(3, '0'), (8, '1')]`, **no match**.
- Continue searching.

```
------v
0001100110
--^
```

- Comparing `[(3, '0'), (7, '0')]`, **match found**.

```
------vv
0001100110
--^^
```

- Comparing `[(4, '1'), (8, '1')]`, **match found**.

```
------vvv
0001100110
--^^^
```

- Comparing `[(5, '1'), (9, '1')]`, **match found**.

```
------vvvv
0001100110
--^^^^
```

- Comparing `[(6, '0'), (10, '0')]`, **match found**.

We reach the **end of the sequence** while matching, so we increase **complexity** by 1.  
Final compressed segments:

```
['000', '11', '00110']
```

---

### **Conclusion**

By applying **Lempel-Ziv compression**, we efficiently detect and encode repeated sequences, leading to the compressed representation:

```
['000', '11', '00110']
```

This process optimizes the data storage while maintaining the structure of the original signal.