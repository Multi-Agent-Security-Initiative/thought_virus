According to "Token Entanglement in Subliminal Learning", the strongest effect in Qwen2.5-7B-Instruct is with penguin (90x), elephant (70x), and kangaroo (26x).
We try whether we can subliminally prompt with the system prompt for these animals and the corresponding reported three-digit numbers (Figure 4, Appendix A).

# Results
Some numbers seem to greatly influence the probability of answering certain animals.
Partly, these numbers are the ones that they discovered in the oriignal paper (e.g. kangaroo and 998, or elephant and 130), but also some of the numbers they discovered don't work well in our setup, and some other numbers perform well (e.g. elephant and 715).

# Learnings
1. Subliminal learning in user prompt seems possible.
2. The numbers that work well may be different ones in user prompt compared to system prompt.
3. The whole thing seems to be pretty noisy, might make sense to crank up # of samples even more (currently 1,000).