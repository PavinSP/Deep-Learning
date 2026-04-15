# Session 3: Deep Networks, Training & Backpropagation — Slide-by-Slide Notes

*Detailed, from-scratch explanations of every slide in Session 3 of Professor Magda Gregorová's Deep Learning course. Assumes zero prior knowledge and builds directly on Session 2.*

---

## Slide 1: "Deep Networks, Training & Backpropagation" — The Title & Context

### What This Session Is About

The title of Session 3 is:

**"Deep Networks, Training & Backpropagation"**

This title is doing a lot of work. Just like Session 2's title told you what kind of model you were about to study, Session 3's title tells you what the next big leap is:

- In **Session 1**, you learned **what** Deep Learning is and why it matters.
- In **Session 2**, you learned **how a shallow neural network works** mathematically.
- In **Session 3**, you move to the next two real questions:
  1. **How do we make networks deeper?**
  2. **How do we actually train them?**

This is the session where the neural network stops being just a static formula on paper and starts becoming a **learnable system**.

---

### Word 1: "Deep"

This is the first key word.

In Session 2, you studied a **shallow neural network**, which means:

```text
Input -> Hidden Layer -> Output
```

That is only **one hidden layer**.

A **deep** network means:

```text
Input -> Hidden 1 -> Hidden 2 -> Hidden 3 -> ... -> Output
```

So "deep" simply refers to the **number of layers of processing** between the input and the output.

#### Why does depth matter?

Because many real-world problems are too complicated to solve efficiently with just one hidden layer.

A deep network can learn in **stages**:

- early layers detect simple patterns,
- middle layers combine them into bigger patterns,
- later layers combine those into meaning.

> **Analogy - Building Understanding Layer by Layer:**
> Imagine recognizing a face.
> - Layer 1 notices edges
> - Layer 2 notices shapes like curves and corners
> - Layer 3 notices eyes, nose, mouth
> - Layer 4 notices the whole face
>
> A shallow network tries to do all of this in one jump.
> A deep network does it step by step.

That step-by-step structure is the heart of deep learning.

---

### Word 2: "Networks"

This word should now feel more familiar because Session 2 already built the intuition.

A **network** means many neurons connected together so that information flows through them.

In Session 2, the network was small:

- one input layer,
- one hidden layer,
- one output layer.

In Session 3, the idea is not changing. The only difference is that we are now **stacking layers**.

So a deep network is not a completely different machine. It is just the same basic building block repeated multiple times.

```text
Session 2:
x -> hidden -> y_hat

Session 3:
x -> hidden 1 -> hidden 2 -> hidden 3 -> ... -> y_hat
```

That is an important conceptual point:

> **Deep networks are not new magic.**
> They are shallow-network building blocks stacked on top of each other.

---

### Word 3: "Training"

This is probably the most important new word in the title.

In Session 2, you learned the form of the network:

$$\hat{y} = f_\theta(x)$$

You learned that:

- `x` is the input,
- `theta` are the parameters,
- `y_hat` is the prediction.

But one huge question was still missing:

> **Where do the values of theta come from?**

Do we choose them by hand?

No.

That is what **training** is.

### What training means

Training is the process of adjusting the parameters `theta` so that the network's predictions get closer and closer to the correct answers.

In plain English:

- the network makes a guess,
- we compare the guess to the truth,
- we measure how wrong it was,
- then we change the parameters a little bit,
- and repeat this many times.

So training is basically:

```text
Guess -> Measure error -> Adjust parameters -> Guess again -> Repeat
```

#### Why is training necessary?

Because a neural network with random parameters is just a random function.

It has the right architecture, but it does not yet "know" anything.

> **Analogy - Tuning a Musical Instrument:**
> A guitar may have the correct shape, strings, and body.
> But if the strings are not tuned, it sounds wrong.
>
> The architecture of the neural network is like the guitar.
> The parameter values are like the tuning.
> Training is the tuning process.

Without training, even a perfect architecture is useless.

---

### Word 4: "Backpropagation"

This is the most technical word in the title, and the one students usually find intimidating at first.

But at a high level, the idea is not mystical.

Backpropagation is the method that tells the network:

- **which parameters caused the error,**
- **how much they contributed,**
- and **how to change them to reduce that error.**

In other words, backpropagation is the system that sends feedback backward through the network so learning can happen.

#### Why is it called "backpropagation"?

Because information moves in two directions during training:

1. **Forward pass**
   - Input goes forward through the layers
   - Network produces an output

2. **Backward pass**
   - Error information moves backward
   - Each parameter gets told how it should change

That backward flow of error information is called **backpropagation**.

> **Analogy - Teacher Correcting an Exam Chain:**
> Imagine a group project where:
> - Student A passes work to Student B
> - Student B passes to Student C
> - Student C gives the final answer
>
> If the final answer is wrong, the teacher has to trace backward:
> - where did the mistake enter?
> - was Student C's combination wrong?
> - did Student B pass incorrect intermediate work?
> - did Student A start from the wrong base?
>
> Backpropagation does exactly that for neural networks.

It traces error backward through the chain of computations.

---

### Why These Three Ideas Belong Together

The title is not just a random list. These ideas form a chain:

1. **Deep Networks**
   - More layers, more expressive power

2. **Training**
   - We need a way to make those layers learn useful parameters

3. **Backpropagation**
   - This is the mechanism that makes that training possible

So the full story of Session 3 is:

```text
We stack more layers -> now we have a deep network
A deep network has many parameters -> we need to train them
To train them efficiently -> we use backpropagation
```

That is the conceptual roadmap hidden inside the title slide.

---

### Why Session 3 Is a Big Turning Point

Session 2 gave you the **structure** of a neural network.

Session 3 gives you the **learning process**.

That is a huge shift.

Before Session 3, a neural network is mostly:

- notation,
- formulas,
- architecture,
- theory.

After Session 3, it becomes:

- a model that makes predictions,
- gets feedback,
- updates itself,
- and gradually improves.

This is where deep learning starts to feel like a real algorithm rather than just a diagram.

---

### Connection to Session 2

Session 2 ended with the question:

> *How do we actually find the best values of theta?*

That unanswered question is exactly what Session 3 is about.

Session 2 taught:
- what a neuron computes,
- what a hidden layer does,
- what a shallow network looks like,
- why neural networks are expressive.

Session 3 now asks:
- how do we stack layers cleanly?
- how do we measure whether predictions are good or bad?
- how do we update parameters?
- how do we compute those updates efficiently?

So Session 3 is the natural continuation, not a separate topic.

---

### The Real Meaning of the Title Slide

The title slide is simple visually, but conceptually it is announcing:

- **architecture gets deeper,**
- **learning becomes central,**
- **mathematics becomes more dynamic,**
- **and backpropagation becomes the engine of optimization.**

If Session 2 was about **what the network is**, then Session 3 is about **how the network learns**.

---

### ✅ Check Your Understanding — Slide 1 Questions

#### Conceptual Questions:

**Q1.** In your own words, what is the difference between a **shallow** neural network and a **deep** neural network?

> **Pavin's Answer:** the difference between a shalow and deep neural netrowk is in shallow neural netowrk there is just a single layer but in deep there are many layers of neurons (hidden neurons)
>
> **✅ Very close!** You have the main idea exactly right: the difference is the **number of hidden layers**.
>
> Small correction for precision:
> - A **shallow neural network** does **not** mean "only one layer total"
> - It means **one hidden layer** between the input and the output
> - A **deep neural network** means **multiple hidden layers**
>
> So the clean version is:
> - **Shallow:** input -> **1 hidden layer** -> output
> - **Deep:** input -> **many hidden layers** -> output
>
> That hidden-layer count is what "depth" refers to.

**Q2.** Why is knowing the architecture of a neural network not enough? Why do we also need **training**?

> **Pavin's Answer:** with architecture we can find the output but when we compare with the actual output if it is wrong we never know how to rectify the mistakes, so we need training where the model will realise there were problems in the weights of the parameters and will learn to update them to find the best solution possible closely accurate to the actual output
>
> **✅ Strong answer!** You identified the key issue: architecture tells us **how the computation is organised**, but not **which parameter values make the computation correct**.
>
> The crucial distinction is:
> - **Architecture** = the shape of the network
> - **Training** = the process of finding good parameter values inside that shape
>
> A network can have the perfect architecture and still fail badly if the weights are poor. Training is what lets the model compare prediction vs truth, measure the error, and gradually adjust the parameters to reduce that error.
>
> A cleaner version of your answer would be:
> > The architecture tells us how the network computes an output, but training is needed to learn the parameter values that make the output close to the correct answer.

**Q3.** At a high level, what do you think **backpropagation** is doing inside the network?

> **Pavin's Answer:** backproparagration first subtracts yhat and y and if erro ris high it will go back to the hideen neurons and will change the neurons weight responsible for this high error and again will calculate the output and again check the error and will repeat this process until a good low error is acheived
>
> **✅ Good intuition, but this needs an important refinement.** You correctly understand the big picture:
> - compare prediction with truth,
> - detect error,
> - send that information backward,
> - update parameters,
> - repeat until error gets smaller.
>
> But there are two technical corrections:
>
> 1. **Backpropagation does not directly "change the weights."**
>    It computes how much each parameter contributed to the error.
>    In more formal terms, it computes the **gradients**.
>
> 2. **The actual weight update is usually done by gradient descent or an optimizer.**
>    So:
>    - **Backpropagation** = computes the feedback signal / gradients
>    - **Gradient descent** = uses those gradients to update the weights
>
> Also, the error is not always just "subtract y_hat and y" directly. Usually we define a **loss function** that measures how bad the prediction is, and backpropagation works backward from that loss.
>
> A more precise high-level version is:
> > Backpropagation sends error information backward through the network and calculates how each parameter should change so that the loss becomes smaller.

#### Application Question:

**Q4.** Imagine you build a deep network for detecting whether an X-ray shows a fracture. Why would random parameter values be useless, even if the architecture is correct?

> **Pavin's Answer:** if it's random then the model will not deted the fracture cuz it will work on its own
>
> **✅ Correct core idea!** Random parameters mean the network has not learned any meaningful pattern yet.
>
> The architecture may be capable of solving the problem, but with random weights:
> - the network has no learned understanding of fracture shapes,
> - no learned edge detectors,
> - no learned medical patterns,
> - and no reason to produce sensible outputs.
>
> So the predictions would mostly be arbitrary or accidental. It might occasionally guess correctly by chance, but it would not be reliably detecting fractures.
>
> A stronger version of your answer would be:
> > Random parameters are useless because the network has not learned the visual patterns that distinguish a fractured X-ray from a healthy one, so its outputs are essentially random guesses.

#### Mini Coding Warm-up:

**Q5.** If a neural network is written as `y_hat = f_theta(x)`, which part is supposed to change during training:
- `x`
- `theta`
- or the definition of the function itself?

> **Pavin's Answer:** f)theta will change cuz the input will not change at all so the function which calculates the weighted sum will and should change
>
> **⚠️ Almost right - the answer is `theta`, but the explanation needs one correction.**
>
> You are absolutely right that **the input `x` does not change** during training for a given example. The part that changes is **`theta`**.
>
> But we should be careful with the phrase "the function itself changes."
>
> In the notation:
>
> $$\hat{y} = f_\theta(x)$$
>
> - **`f`** = the model form / architecture / rule of computation
> - **`theta`** = the parameter values inside that model
> - **`x`** = the input data
>
> During training:
> - `x` stays the input
> - the **definition of the model family stays the same**
> - the **parameter values `theta` change**
>
> So the most precise answer is:
> > **`theta` changes during training.** The architecture and computation rule stay the same, but the parameter values inside that rule get updated.

---

## Slide 2: Lecture Overview — The Roadmap of Session 3

### What's On This Slide

This slide is short, but it is extremely important. It gives you the **entire roadmap** of Session 3 in just four lines:

1. **Deep Networks & Notation**
2. **The Learning Problem**
3. **Loss Functions**
4. **Gradient Descent**

At first glance, this may look like a boring table of contents. But it is actually telling you the full *logic* of the session:

- first, we build the deeper model,
- then, we define what problem we are trying to solve,
- then, we define how to measure whether we are doing well or badly,
- and finally, we learn how to improve the model step by step.

So this slide is not just saying **what** topics will appear. It is also saying **why they must appear in this order**.

---

### Why Session 3 Needs a Roadmap

Session 2 already gave you the basic neural network building block:

```text
input -> hidden layer -> output
```

But Session 3 is more complex than Session 2 because now we are moving into:

- multiple layers,
- more notation,
- training objectives,
- error measurement,
- optimization.

That means Session 3 is no longer just about one formula. It is about a **whole learning pipeline**.

This roadmap slide is the professor's way of saying:

> "Don't get lost. Here is the path we are going to follow."

---

### Topic 1: Deep Networks & Notation

This is the first topic for a reason.

Before we can train anything, we need to define **what the model actually is**.

In Session 2, the model was shallow:

```text
x -> hidden -> y_hat
```

In Session 3, we move to:

```text
x -> hidden 1 -> hidden 2 -> hidden 3 -> ... -> y_hat
```

That means two things happen at the same time:

1. The network becomes **deeper**
2. The notation becomes **harder to manage**

Why does notation suddenly matter so much?

Because with one hidden layer, you can still write formulas neuron by neuron. But once you stack many layers, the equations quickly become messy:

- different layers need different symbols,
- each layer has its own weights and biases,
- outputs of one layer become inputs to the next,
- and writing everything with separate sums becomes painful.

So Topic 1 is not just about "deep networks." It is also about inventing a **clean language** for describing them.

> **Analogy - Upgrading from a Sketch to a Blueprint:**
> A simple house can be explained with a rough sketch.
> A skyscraper cannot.
>
> Once the structure gets large, you need proper notation the way an architect needs a proper blueprint.

That is exactly what this first part of the session will do.

---

### Topic 2: The Learning Problem

Once the model is defined, the next question is:

> **What are we actually trying to learn?**

This is the **learning problem**.

At a high level, the learning problem is:

- we have data,
- we have a model with parameters,
- we want to choose parameter values that make the model perform well.

In Session 2, you already saw the idea:

$$\hat{y} = f_\theta(x)$$

Now the learning problem asks:

> **How do we choose theta so that `f_theta(x)` gives good predictions on real data?**

This is where deep learning becomes an optimization problem rather than just an architecture diagram.

The learning problem includes questions like:

- What data do we train on?
- What does "good prediction" mean?
- Are we trying to fit the training data only, or also generalize to new data?
- What exactly counts as a better set of parameters?

So Topic 2 gives the session its main objective.

---

### Topic 3: Loss Functions

Once we say "we want the model to perform well," we immediately hit a problem:

**How do we measure "well"?**

That is what a **loss function** does.

A loss function takes the model's prediction and the true answer and produces a number that says:

- small number = good,
- large number = bad.

Without a loss function, training is impossible, because the model would have no precise signal telling it whether it is improving.

#### Why do we need a loss instead of just saying "right" or "wrong"?

Because learning usually happens gradually.

Suppose the true value is 10:
- prediction = 9.9
- prediction = 6
- prediction = -100

All three are "wrong" in a strict sense.
But clearly they are **not equally wrong**.

A loss function gives us a way to measure *how wrong* the prediction is.

> **Analogy - A Report Card Score:**
> If a teacher only said "wrong" for every mistake, you would not know whether you were improving from 20% to 60% to 90%.
>
> A score gives you a direction.
> A loss function does the same thing for the network.

So Topic 3 gives the model a **numerical learning signal**.

---

### Topic 4: Gradient Descent

Now we finally have:

- a model,
- a learning objective,
- and a loss function that tells us how bad we are.

The next question is:

> **How do we actually improve the parameters?**

That is where **gradient descent** comes in.

Gradient descent is the algorithmic rule that says:

- look at how the loss changes with respect to the parameters,
- figure out which direction makes the loss smaller,
- move a little in that direction,
- repeat many times.

So if Topic 3 tells us **how bad** the model is, Topic 4 tells us **how to make it better**.

> **Analogy - Walking Downhill in Fog:**
> Imagine you are standing on a mountain in thick fog and want to reach the bottom.
> You cannot see the whole landscape.
> But you can feel the slope under your feet.
>
> If you repeatedly step in the downhill direction, you gradually descend.
>
> Gradient descent does exactly that in parameter space.

This is the heart of training.

---

### Why These Four Topics Are in This Order

The ordering is not random. It forms a chain:

```text
1. Deep Networks & Notation
   "What model are we working with?"

2. The Learning Problem
   "What are we trying to achieve with that model?"

3. Loss Functions
   "How do we measure whether we are doing badly or well?"

4. Gradient Descent
   "How do we update the model to reduce that badness?"
```

This is the hidden logic of the session:

```text
Model -> Objective -> Error Measure -> Improvement Rule
```

If you skip any one of these, the whole learning story breaks:

- without the model, there is nothing to train,
- without the learning problem, there is no goal,
- without the loss, there is no measurement,
- without gradient descent, there is no update mechanism.

---

### Where Does Backpropagation Fit In?

A very important observation:

The session title includes **Backpropagation**, but the roadmap slide does **not** list it as a separate top-level bullet.

Why?

Because backpropagation is not really a separate destination. It is the **mechanism inside the training process** that makes gradient-based learning possible.

In other words:

- **Loss function** tells us what to minimize
- **Backpropagation** computes how each parameter affects that loss
- **Gradient descent** uses that information to update parameters

So backpropagation sits in the middle of the learning machinery. It is part of how Topic 3 connects to Topic 4.

That is why the title highlights it, even though the roadmap organizes the material more broadly.

---

### The Story Arc of Session 3

If we compress the whole lecture roadmap into plain English, it becomes:

```text
Step 1: Build a deeper network and create notation clean enough to describe it
Step 2: State the learning task formally
Step 3: Define a loss that measures prediction error
Step 4: Use gradients to reduce that loss and improve the model
```

This is the session where deep learning becomes a true **optimization pipeline**.

Session 2 asked:

> "What is a neural network?"

Session 3 asks:

> "How do we make a deep neural network learn?"

That is the big shift.

---

### Why This Slide Matters for the Rest of the Session

This slide quietly tells you what to pay attention to in the rest of the lecture:

1. Whenever the professor introduces new notation, it belongs to **Topic 1**
2. Whenever she defines what training is trying to achieve, it belongs to **Topic 2**
3. Whenever she compares prediction to truth numerically, it belongs to **Topic 3**
4. Whenever she explains how parameters are updated, it belongs to **Topic 4**

So this overview slide is like a map legend. It tells you how to categorize every later idea.

---

### ✅ Check Your Understanding — Slide 2 Questions

#### Conceptual Questions:

**Q1.** Why does Session 3 start with **Deep Networks & Notation** instead of jumping straight to training?

**Q2.** In your own words, what is the difference between the **learning problem** and the **loss function**?

**Q3.** Why do we need a loss function before we can use gradient descent?

#### Application Question:

**Q4.** Suppose you build a neural network and it gives two different predictions on the same example under two different parameter settings. Which topic from this roadmap tells you how to decide which setting is better?

#### Mini Coding Warm-up:

**Q5.** In a training loop, which step corresponds most directly to **loss functions**, and which step corresponds most directly to **gradient descent**?

For example, think about this sequence:

```python
y_hat = model(x)
loss = compute_loss(y_hat, y)
# use gradients somehow
# update parameters somehow
```

Which line belongs to Topic 3, and which later step belongs to Topic 4?
