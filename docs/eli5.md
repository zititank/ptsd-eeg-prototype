# Explain it like I'm five

*Written by Bogdan (zititank).*

## What is EEG, and what did we record?

Your brain cells talk to each other with tiny electrical signals. An EEG puts sensors on your head and listens to
the sum of all that chatter from outside the skull. Our headset had six dry sensors: two at the back of the head,
two at the forehead and two at the sides. Each person sat still with their eyes closed for about a minute. We had
about 90 such recordings for training, 24 of them from people with PTSD.

## What is the alpha rhythm, and why did we look at it?

Close your eyes and relax, and the back of your brain starts to hum at about 10 beats per second. That hum is the
alpha rhythm, and it's the easiest thing to see in EEG. Research has linked alpha to stress and arousal, so it was
the obvious place to look. It also has a useful side effect: if a recording with closed eyes has no alpha at all,
something is probably wrong with the recording.

## What did the model learn?

It learned one rule, "more alpha means more likely PTSD". The main weight is positive in 98.7% of the 9000 small
models we trained. The problem is that research says PTSD comes with the same or *less* alpha, not more. So the
model wasn't learning PTSD. It was learning the difference between "a recording that has alpha" and "a recording
that doesn't". That turned out to be the difference between real brain signal and broken files.

## Why is the big number (AUC 0.816) misleading?

AUC measures how well a model ranks sick people above healthy ones: 1.0 is perfect and 0.5 is a coin flip. We got
0.816, which looks good. But 46 of our 66 healthy controls came from one batch where not a single recording had an
alpha peak. That batch is easy to tell apart from anything, so against it the AUC is 0.982. Against the 20 controls whose recordings
were real, the AUC is 0.435, a coin flip. Even worse, a model that saw only the file metadata (no brain signal at
all) scored 0.97. The big number was measuring broken recordings.

## How did we make sure we weren't fooling ourselves?

- **Leave one person out.** The model never sees the person it is tested on. We did this 90 times, once per person,
  and even the model's settings were chosen without looking at that person.
- **Rules before results.** Before trying new models, we wrote down how the winner would be chosen and saved a
  fingerprint (a SHA-256 hash) of that document. Nobody could change the rules after seeing the scores.
- **Leakage tests.** If you shuffle the labels, a leaky pipeline still "finds" something. Ours scored 0.349, inside
  the range you'd expect by pure chance.
- **Reproducibility.** Retraining from scratch gives the same scores to within 5e-16, which is rounding noise.
- **The biology check.** When the main weight pointed the wrong way, we treated that as a bug report, not a
  discovery.

## What would a real PTSD brain test need?

- Every group recorded the same way: same device, same room, same protocol, same period of time.
- An "is it brain?" check on every file before anyone looks at who has PTSD.
- Many more people, from more than one place, and testing on a site the model has never seen.
- Information about age, medication and sleep, which change EEG on their own.
- A clinical trial before it goes anywhere near a patient. This project is a prototype and is not that.

## What I learned

I went in thinking the hard part was the model. The hard part was the data, and the useful part of what I built was
the set of safety nets around the model. Don't assume data is universal: a file from one batch isn't the same
thing as a file from another, even if the format matches. Biology can catch what statistics misses, and
statistics can prove what biology suspects. And when AI writes most of your code, your job becomes deciding what
to check. The long version is in [story.md](story.md).
