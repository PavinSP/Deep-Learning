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

> **Pavin's Answer:** only if we understand how to do notation and understand the completx layering of hidden neurons will we understand training proccess
>
> **✅ Correct idea.** Session 3 starts with **Deep Networks & Notation** first because training only makes sense once we clearly understand:
> - what the network structure looks like,
> - how many hidden layers it has,
> - and how to write that structure in a clean mathematical way.
>
> If the network becomes deep, the computations across layers become more complex. Without proper notation, it becomes very hard to describe:
> - what each layer is doing,
> - how information moves forward,
> - and later how error information moves backward.
>
> So Topic 1 comes first because we need a clear language for the model before we can explain how that model is trained.
>
> A cleaner version of your answer would be:
> > We need Deep Networks & Notation first because training a deep model is hard to explain unless we already understand the layered structure of the network and the notation used to describe it.

**Q2.** In your own words, what is the difference between the **learning problem** and the **loss function**?

> **Pavin's Answer:** learning problem is defining the objective what we should get as output and loss function is the difference in actual output and predicted output
>
> **✅ Good answer.** You correctly separated the two ideas:
>
> - The **learning problem** asks: what are we trying to make the model achieve?
> - The **loss function** asks: how do we measure how far the model's current prediction is from that goal?
>
> So the learning problem is about the **target of learning**, while the loss function is about the **numerical measurement of error**.
>
> One small refinement: a loss function is not always just the simple difference between actual output and predicted output. More generally, it is a rule that converts prediction vs truth into a number telling us how bad the prediction is.
>
> A cleaner version of your answer would be:
> > The learning problem defines the objective of training, while the loss function measures how wrong the model's prediction is compared with the true output.

**Q3.** Why do we need a loss function before we can use gradient descent?

> **Pavin's Answer:** because only if can define how much erro we have can we do gradient descent ie optmize the parameters or we can't define the loss
>
> **✅ Yes, that is the main reason.** Gradient descent needs a numerical quantity to minimize. That quantity is the **loss**.
>
> Without a loss function:
> - we do not know how wrong the prediction is,
> - we do not have a precise objective to optimize,
> - and gradient descent has no clear signal telling it which direction should reduce the error.
>
> So the loss function comes first because it defines the surface that gradient descent is trying to move down.
>
> A cleaner version of your answer would be:
> > We need a loss function before gradient descent because gradient descent can only optimize the parameters if we have a numerical measure of error to minimize.

#### Application Question:

**Q4.** Suppose you build a neural network and it gives two different predictions on the same example under two different parameter settings. Which topic from this roadmap tells you how to decide which setting is better?

> **Pavin's Answer:** we should calculate loss function for botha nd then see which one has less
>
> **✅ Correct.** This belongs to **Topic 3: Loss Functions**.
>
> If two parameter settings give two different predictions, we compare them by computing the loss for each one on the same example. The setting with the **smaller loss** is considered better because it produced a prediction closer to the true answer.
>
> This is exactly why Topic 3 matters: it gives us a principled way to compare outputs numerically instead of judging them vaguely.
>
> A cleaner version of your answer would be:
> > We use Topic 3, Loss Functions, by computing the loss for both predictions and choosing the parameter setting with the smaller loss.

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

> **Pavin's Answer:** line 3 belongs to topic 3 and line 4 belongs to topic 4
>
> **⚠️ Close, but one correction is needed.**
>
> In the code snippet:
>
> ```python
> y_hat = model(x)
> loss = compute_loss(y_hat, y)
> # use gradients somehow
> # update parameters somehow
> ```
>
> - `loss = compute_loss(y_hat, y)` corresponds most directly to **Topic 3: Loss Functions**
> - `# update parameters somehow` corresponds most directly to **Topic 4: Gradient Descent**
>
> So Topic 3 is the line where the loss is actually computed, and Topic 4 is the later step where the parameters are updated to reduce that loss.
>
> The line `# use gradients somehow` sits in between and is closer to the gradient/backpropagation mechanism that helps make the update possible.
>
> A cleaner version of your answer would be:
> > The loss-computation line belongs to Topic 3, and the parameter-update step belongs to Topic 4.

---

## Slide 3: From Shallow to Deep

### What's On This Slide

This is the first slide where Session 3 starts building the actual idea of a **deep** neural network.

The slide does four things at once:

1. It reminds you of the **1-hidden-layer network** from Session 2
2. It shows the new idea: **stack more layers**
3. It states the repeating pattern inside each hidden layer
4. It explains why we now need **cleaner notation**

So this slide is the bridge between:

- the shallow network you already understand,
- and the deeper architecture that the rest of Session 3 will formalize.

The key message is:

> A deep network is not a completely different machine.
> It is the same basic neural-network computation repeated layer after layer.

---

### Recall: The 1-Hidden-Layer Network

The left side of the slide recalls the shallow network from Session 2:

$$
h_j = a\left(\theta_{j0} + \sum_i \theta_{ji} x_i\right)
$$

$$
\hat{y} = \theta_0 + \sum_j \theta_j h_j
$$

This is the standard 1-hidden-layer story:

- the input is `x = (x_1, x_2, ..., x_d)`,
- each hidden neuron computes a weighted sum of the inputs,
- then applies an activation function `a(.)`,
- and the output combines all hidden activations into the prediction `\hat{y}`.

Let us unpack the first equation carefully:

$$
h_j = a\left(\theta_{j0} + \sum_i \theta_{ji} x_i\right)
$$

This means:

- `h_j` = hidden neuron number `j`
- `x_i` = input feature number `i`
- `\theta_{ji}` = weight connecting input `x_i` to hidden neuron `h_j`
- `\theta_{j0}` = bias of hidden neuron `j`
- `a(.)` = activation function

So hidden neuron `j` works in two steps:

1. Compute a weighted sum plus bias
2. Pass that result through a nonlinear activation

Then the network forms the final output:

$$
\hat{y} = \theta_0 + \sum_j \theta_j h_j
$$

That equation says:

- take all hidden activations,
- weight them,
- add an output bias,
- and produce the final prediction.

This is exactly the model you already know from Session 2.

---

### What the Diagram on the Left Is Showing

The picture on the left is the visual version of those equations.

It shows:

- an **input layer** with features `x_1, x_2, ..., x_d`,
- one **hidden layer** with neurons `h_1, h_2, ..., h_k`,
- and one **output node** producing `\hat{y}`.

Important details:

- `d` is the number of input features
- `k` is the number of hidden neurons
- every arrow represents a **learnable parameter**

So when you see many arrows from the inputs into one hidden neuron, that means:

> each hidden neuron looks at **all input features**, but with its own learned weights.

That is why different hidden neurons can learn different patterns from the same input.

---

### The New Idea: Stack More Layers

The right side of the slide introduces the big step:

```text
input -> hidden 1 -> hidden 2 -> output
```

This is what makes the network **deep**.

Instead of going directly from:

```text
input -> hidden -> output
```

we now allow:

```text
input -> hidden 1 -> hidden 2 -> hidden 3 -> ... -> output
```

That means the output of one hidden layer becomes the input to the next hidden layer.

So the network no longer performs just one nonlinear transformation before the output. It performs a **sequence of transformations**.

This is the real meaning of depth:

- first layer transforms the raw input,
- second layer transforms the first layer's representation,
- third layer transforms that again,
- and so on until the output is produced.

So a deep network is best understood as a **chain of learned representations**.

---

### The Hidden Mathematical Idea: Composition

Another way to describe the same idea is:

> A deep network is a composition of functions.

A shallow network looks roughly like:

```text
x -> hidden representation -> output
```

A deep network looks more like:

```text
x -> f1(x) -> f2(f1(x)) -> f3(f2(f1(x))) -> ... -> y_hat
```

Each layer takes the previous layer's output and transforms it again.

This matters because many real problems are easier to solve in stages.

For example:

- an early layer may detect simple patterns,
- a middle layer may combine them into more meaningful structures,
- a later layer may combine those into task-relevant concepts.

This is why deep networks are powerful: they can build understanding gradually rather than trying to do everything in one jump.

---

### The Core Pattern: Linear Transform + Nonlinear Activation

The slide states:

> **Pattern: each layer = linear transform + nonlinear activation**

This sentence is one of the most important ideas in all of deep learning.

Every hidden layer follows the same recipe:

1. **Linear transform**
   - multiply inputs by weights
   - add biases

2. **Nonlinear activation**
   - apply an activation function such as ReLU, sigmoid, or tanh

In simple symbolic form, one layer does:

```text
z = Wv + b
h = a(z)
```

where:

- `v` is the input coming into that layer,
- `W` is the weight matrix,
- `b` is the bias vector,
- `z` is the pre-activation linear result,
- `a(.)` is the activation function,
- `h` is the layer output.

So each layer is doing the same type of operation, just on a different input representation.

This repetition is why deep networks are conceptually simple even when they become large.

---

### Why the Nonlinearity Matters

The word **nonlinear** is not decoration. It is essential.

If every layer only did a linear transform and there were **no activation function**, then stacking many layers would not really give you a fundamentally richer model.

Why?

Because a sequence of linear transformations can be collapsed into one bigger linear transformation.

In plain language:

- linear layer after linear layer after linear layer
- is still just one overall linear mapping

So the network would gain size, but not real expressive depth.

The activation function is what prevents that collapse. It lets the model represent more complicated input-output relationships.

That is why the slide emphasizes:

```text
linear transform + nonlinear activation
```

and not just:

```text
linear transform
```

---

### What Actually Changes When We Go Deep?

When we move from shallow to deep, several things change at once:

1. **The number of layers increases**
   - We now have hidden layer 1, hidden layer 2, and possibly many more

2. **The number of parameters increases**
   - Every new layer brings new weights and biases

3. **The intermediate representations become more abstract**
   - Later layers no longer process raw input directly
   - They process the features created by earlier layers

4. **The formulas become harder to write**
   - We now need indices for both neurons and layers

That last point is the reason this slide ends by saying:

> **We need cleaner notation to handle this.**

---

### Why Scalar Notation Starts Getting Messy

With one hidden layer, scalar notation is still manageable:

$$
h_j = a\left(\theta_{j0} + \sum_i \theta_{ji} x_i\right)
$$

But imagine writing a 2-hidden-layer network the same way:

$$
h^{(1)}_j = a\left(\theta^{(1)}_{j0} + \sum_i \theta^{(1)}_{ji} x_i\right)
$$

$$
h^{(2)}_m = a\left(\theta^{(2)}_{m0} + \sum_j \theta^{(2)}_{mj} h^{(1)}_j\right)
$$

$$
\hat{y} = \theta^{(3)}_0 + \sum_m \theta^{(3)}_m h^{(2)}_m
$$

This is already much harder to read.

Now imagine 5 layers, 10 layers, or vector outputs.

Very quickly, scalar-by-scalar notation becomes:

- long,
- cluttered,
- easy to confuse,
- and hard to manipulate mathematically.

So Slide 3 is preparing you for the next logical step:

> use **matrix notation** and **layer notation** so the deep network can be written cleanly.

That is exactly what the next slides will do.

---

### A Good Way to Mentally Picture Depth

A useful mental model is:

> each hidden layer rewrites the input into a new internal language that is more useful for the next layer.

So:

- the first hidden layer does not solve the whole problem,
- the second hidden layer does not solve the whole problem,
- the output layer does not start from raw input,
- instead, each layer passes along a more processed representation.

You can think of it as a pipeline:

```text
raw input
-> first interpretation
-> more refined interpretation
-> even more task-specific interpretation
-> final prediction
```

That is the intuition behind hierarchical representation learning.

---

### Why This Slide Matters for the Rest of Session 3

This slide is more important than it looks because it sets up almost everything that follows:

- Slide 4 will introduce **matrix notation**
- Slide 5 will introduce **layer notation**
- later slides will define the **learning problem**
- then the course will explain **loss functions**
- and finally **gradient descent** and **backpropagation**

So Slide 3 is the conceptual doorway into the rest of the session.

It tells you:

- what a deep network is,
- what repeated structure all layers share,
- and why the mathematics now needs to become more compact.

---

### Main Takeaway

The whole slide can be compressed into one sentence:

> A deep neural network is built by stacking the same kind of layer repeatedly, where each hidden layer performs a linear transform followed by a nonlinear activation.

And the final line of the slide adds the practical consequence:

> once we stack many layers, we need better notation to describe the network cleanly.

---

## Slide 4: Matrix Notation

### What's On This Slide

This slide takes the shallow-network equation from Session 2 and rewrites it in a much more compact form using **vectors and matrices**.

That is a major step because once networks become deep, writing every neuron separately becomes too messy.

So Slide 4 is really answering the question raised at the end of Slide 3:

> **How do we describe one whole layer cleanly instead of one neuron at a time?**

The answer is:

- collect all weights into a matrix,
- collect all hidden activations into a vector,
- collect all biases into a vector,
- and write the whole layer computation in one line.

---

### The Scalar Equation We Are Starting From

The slide begins with the familiar scalar notation:

$$
h_j = a\left(\theta_{j0} + \sum_{i=1}^{d} \theta_{ji} x_i\right), \quad j = 1, \dots, k
$$

This is the same hidden-layer formula from Session 2.

It says:

- we have `d` input features,
- we have `k` hidden neurons,
- and for each hidden neuron `j`, we compute one weighted sum of the inputs and then apply the activation function.

Important symbols here:

- `x_i` = input feature `i`
- `h_j` = hidden neuron `j`
- `\theta_{ji}` = weight from input `i` to hidden neuron `j`
- `\theta_{j0}` = bias of hidden neuron `j`
- `a(.)` = activation function

So this equation is perfectly correct, but it has a problem:

> it only tells us how **one neuron** works at a time.

If a layer has many neurons, we have to keep writing this same expression over and over for `j = 1, ..., k`.

That is exactly what matrix notation fixes.

---

### Why Scalar Notation Starts Becoming Uncomfortable

With one neuron, scalar notation is natural.

With one hidden layer of many neurons, it is still manageable.

But once we move toward deep networks, scalar notation becomes awkward because:

- every neuron has its own index,
- every layer has its own index,
- every connection has its own weight,
- and every formula gets longer and harder to read.

So the goal is not to change the mathematics. The goal is to write the **same mathematics more cleanly**.

That is why Slide 4 is about notation, not about a new type of network.

---

### Collecting the Weights Into a Matrix

The slide defines the weight matrix:

$$
\Theta =
\begin{pmatrix}
\theta_{11} & \cdots & \theta_{1d} \\
\vdots & \ddots & \vdots \\
\theta_{k1} & \cdots & \theta_{kd}
\end{pmatrix}
\in \mathbb{R}^{k \times d}
$$

and the bias vector:

$$
\theta_0 =
\begin{pmatrix}
\theta_{10} \\
\vdots \\
\theta_{k0}
\end{pmatrix}
\in \mathbb{R}^{k}
$$

This is the key idea:

- instead of storing weights neuron by neuron,
- we put **all layer weights together** in one matrix `\Theta`,
- and we put **all hidden biases together** in one vector `\theta_0`.

So now:

- one row of `\Theta` belongs to one hidden neuron,
- one entry of `\theta_0` belongs to one hidden neuron,
- and the whole layer can be written as one vector computation.

---

### What `\mathbb{R}^{k \times d}` and `\mathbb{R}^k` Mean

The notation:

$$
\Theta \in \mathbb{R}^{k \times d}
$$

means:

- `\Theta` is a real-valued matrix
- with `k` rows
- and `d` columns

That matches the layer structure:

- `d` input features come into the layer
- `k` hidden neurons come out of the layer

Similarly,

$$
\theta_0 \in \mathbb{R}^k
$$

means:

- `\theta_0` is a real-valued vector
- with one entry for each of the `k` hidden neurons

So there is:

- one bias per hidden neuron,
- not one bias per input feature.

---

### The Matrix Form of the Whole Layer

Once the weights and biases are grouped together, the entire hidden layer can be written as:

$$
h = a(\theta_0 + \Theta x) \in \mathbb{R}^k
$$

This is the central equation of the slide.

It says:

1. Take the input vector `x`
2. Multiply by the weight matrix `\Theta`
3. Add the bias vector `\theta_0`
4. Apply the activation function `a`
5. Get the hidden activation vector `h`

So instead of writing `h_1, h_2, ..., h_k` separately, we write one vector:

$$
h =
\begin{pmatrix}
h_1 \\
\vdots \\
h_k
\end{pmatrix}
$$

This is much cleaner because the entire layer is now treated as one object.

---

### Why This Matrix Equation Is Exactly the Same as the Scalar One

This is an important point:

> matrix notation is not changing the computation.
> It is only packaging it more efficiently.

To see that, look at the product `\Theta x`.

If:

$$
x =
\begin{pmatrix}
x_1 \\
\vdots \\
x_d
\end{pmatrix}
$$

then the `j`th entry of `\Theta x` is:

$$
(\Theta x)_j = \sum_{i=1}^{d} \theta_{ji} x_i
$$

Then the `j`th entry of `\theta_0 + \Theta x` is:

$$
\theta_{j0} + \sum_{i=1}^{d} \theta_{ji} x_i
$$

And after applying the activation function elementwise, we get:

$$
h_j = a\left(\theta_{j0} + \sum_{i=1}^{d} \theta_{ji} x_i\right)
$$

which is exactly the original scalar formula.

So the matrix equation and scalar equation are mathematically identical.

---

### Understanding the Shapes on the Slide

The slide labels the dimensions explicitly:

- `h` is `k x 1`
- `\theta_0` is `k x 1`
- `\Theta` is `k x d`
- `x` is `d x 1`

This is a very useful habit because it lets you check whether the equation makes sense.

Let us verify it:

$$
\Theta x : (k \times d)(d \times 1) = k \times 1
$$

So `\Theta x` is a `k x 1` vector.

Then:

$$
\theta_0 + \Theta x
$$

is valid because both are `k x 1` vectors.

Finally, applying `a(.)` elementwise keeps the shape unchanged, so:

$$
h \in \mathbb{R}^k
$$

This dimension-checking habit becomes extremely important later when networks have many layers.

---

### Why Each Row of `\Theta` Is the Weights of One Neuron

The slide says:

> **Each row of `\Theta` = weights of one neuron**

This is true because when you compute `\Theta x`, each row of `\Theta` takes a dot product with the input vector `x`.

For example, the first row:

$$
(\theta_{11}, \theta_{12}, \dots, \theta_{1d})
$$

produces the pre-activation input to hidden neuron `1`.

The second row produces the pre-activation input to hidden neuron `2`.

And so on.

So row `j` contains exactly the weights that neuron `j` uses to look at the input features.

That is why rows correspond to neurons.

---

### Why Each Column of `\Theta` Is the Weights From One Input

The slide also says:

> **Each column of `\Theta` = weights from one input**

This is also important.

Take column `i` of `\Theta`:

$$
\begin{pmatrix}
\theta_{1i} \\
\theta_{2i} \\
\vdots \\
\theta_{ki}
\end{pmatrix}
$$

This column tells you how input feature `x_i` connects to **all hidden neurons**.

So:

- row view = "what weights does one neuron use?"
- column view = "where does one input feature send its influence?"

Both interpretations are useful, depending on what part of the network you are thinking about.

---

### What "Activation Applied Elementwise" Means

The slide explicitly says:

> **activation `a` applied elementwise**

This means we do **not** feed the whole vector into some mysterious new function that mixes all coordinates together.

Instead, if:

$$
z = \theta_0 + \Theta x =
\begin{pmatrix}
z_1 \\
z_2 \\
\vdots \\
z_k
\end{pmatrix}
$$

then:

$$
a(z) =
\begin{pmatrix}
a(z_1) \\
a(z_2) \\
\vdots \\
a(z_k)
\end{pmatrix}
$$

So the same activation rule is applied separately to each neuron's pre-activation value.

For example, if `a` were ReLU and:

$$
z =
\begin{pmatrix}
2 \\
-1 \\
0.5
\end{pmatrix}
$$

then:

$$
a(z) =
\begin{pmatrix}
2 \\
0 \\
0.5
\end{pmatrix}
$$

because ReLU keeps positive values and turns negative values into zero.

This is what "elementwise" means.

---

### A Helpful Intermediate Quantity: Pre-Activation

The slide writes:

$$
h = a(\theta_0 + \Theta x)
$$

It is often useful to name the inside part:

$$
z = \theta_0 + \Theta x
$$

Then the layer computation becomes:

$$
h = a(z)
$$

Why is this useful?

Because it separates the two parts of the layer:

1. **Linear step:** `z = \theta_0 + \Theta x`
2. **Nonlinear step:** `h = a(z)`

This two-step view becomes very important later for backpropagation, because the derivatives through the linear part and activation part are handled differently.

---

### Why Matrix Notation Matters Beyond Looking Elegant

Matrix notation is not just cleaner for lecture slides. It matters in practice for at least four reasons:

1. **Compactness**
   - one equation describes the whole layer

2. **Clarity**
   - layer structure becomes easier to see

3. **Implementation**
   - programming libraries naturally use vectors and matrices

4. **Scalability**
   - once one layer is written this way, deep networks are just repetitions of the same pattern

In code, this slide is very close to writing:

```python
h = activation(theta_0 + Theta @ x)
```

So Slide 4 is the point where the lecture's math starts looking much more like actual neural-network code.

---

### How This Connects to Deep Networks

Slide 3 said:

> we can stack more layers

Slide 4 now gives us the language for doing that cleanly.

Once one hidden layer is written as:

$$
h = a(\theta_0 + \Theta x)
$$

we can reuse the same pattern for the next layer, and the next one after that.

So Slide 4 is not just about one shallow layer.

It is preparing the reusable building block for **all later layers** in a deep network.

---

### Main Takeaway

The entire slide can be summarized as:

> Instead of describing each hidden neuron separately, we collect all weights into a matrix and all hidden activations into a vector, so one whole hidden layer can be written compactly as `h = a(\theta_0 + \Theta x)`.

That single equation is the bridge from:

- neuron-by-neuron thinking
- to layer-by-layer thinking

and that shift is essential for understanding deep networks.

---

## Slide 5: Layer Notation

### What's On This Slide

This slide takes the matrix notation from Slide 4 and makes it usable for **many layers**.

Slide 4 told us how to describe one hidden layer:

$$
h = a(\theta_0 + \Theta x)
$$

But a deep network has several layers, so now we need notation that tells us:

- which layer we are talking about,
- what goes into that layer,
- what comes out of that layer,
- and which parameters belong to that layer.

That is exactly what Slide 5 introduces.

The big idea is:

> once networks become deep, we stop thinking only in terms of individual neurons or one isolated layer, and start thinking of the network as a sequence of indexed layers.

---

### The Layer Index `l = 0, 1, ..., L`

The slide says:

$$
\text{Index layers } l = 0, 1, \dots, L
$$

This means we number the layers from the beginning of the network to the end.

The indexing convention is:

- `l = 0` is the **input layer**
- `l = 1, 2, ..., L - 1` are the **hidden layers**
- `l = L` is the **output layer**

So `L` is the index of the final layer, not the number of hidden layers alone.

That is an important detail.

If a network has:

- one input layer,
- two hidden layers,
- and one output layer,

then the indexing would be:

```text
l = 0   input
l = 1   hidden 1
l = 2   hidden 2
l = 3   output
```

So in that case, `L = 3`.

This convention gives us one consistent way to talk about the whole network.

---

### Input Layer: `h^(0) = x`

The slide defines the input layer as:

$$
h^{(0)} = x
$$

This is a very useful notational choice.

It says:

- the input vector `x` can also be viewed as the "activation" of layer 0
- so the network starts with `h^{(0)}`
- and every later layer is built from the output of the previous one

This is elegant because now the whole network looks uniform:

- layer 1 takes `h^{(0)}` as input,
- layer 2 takes `h^{(1)}` as input,
- layer 3 takes `h^{(2)}` as input,
- and so on.

So even though the input layer is not computed using weights and biases, we still give it the same style of symbol as the later layers.

That makes the notation cleaner.

---

### Hidden Layers: The General Rule

For hidden layers, the slide writes:

$$
h^{(l)} = a\left(\theta_0^{(l)} + \Theta^{(l)} h^{(l-1)}\right), \quad l = 1, \dots, L - 1
$$

This is one of the most important equations in the early part of the course.

It says:

- the output of layer `l` is `h^{(l)}`
- it depends on the output of the previous layer `h^{(l-1)}`
- the parameters used at this layer are `\Theta^{(l)}` and `\theta_0^{(l)}`
- and after the linear computation, we apply the activation function `a`

So this is just the Slide 4 equation reused in a deeper-network setting.

The only new ingredient is the superscript `(l)`, which tells us **which layer** each quantity belongs to.

---

### What the Superscript `(l)` Means

This is worth slowing down for because it can be confusing at first.

In:

$$
h^{(l)}, \quad \Theta^{(l)}, \quad \theta_0^{(l)}
$$

the superscript `(l)` does **not** mean "raise to the power `l`."

It is just a **label** saying:

- `h^{(l)}` = activation/output of layer `l`
- `\Theta^{(l)}` = weight matrix of layer `l`
- `\theta_0^{(l)}` = bias vector of layer `l`

So the superscript is an index, not an exponent.

That distinction matters a lot because neural-network notation uses superscripts for labeling layers all the time.

---

### Reading the Hidden-Layer Equation Step by Step

Let us read the hidden-layer formula very literally:

$$
h^{(l)} = a\left(\theta_0^{(l)} + \Theta^{(l)} h^{(l-1)}\right)
$$

Step 1:

- start from the previous layer output `h^{(l-1)}`

Step 2:

- multiply it by the current layer's weight matrix `\Theta^{(l)}`

Step 3:

- add the current layer's bias vector `\theta_0^{(l)}`

Step 4:

- apply the activation function `a`

Step 5:

- call the result `h^{(l)}`

So each hidden layer consumes the previous representation and produces a new representation.

That is the layer-by-layer view of deep learning.

---

### The Output Layer Is Written Separately

The slide then gives a separate formula for the output layer:

$$
\hat{y} = \theta_0^{(L)} + \Theta^{(L)} h^{(L-1)}
$$

Notice something important:

- the hidden-layer formula includes the activation function `a(.)`
- the output-layer formula on this slide does **not**

Why?

Because the output layer depends on the task.

In some problems:

- we may leave the output linear,
- in others we may apply sigmoid,
- in others softmax,
- and in others something else.

So at this stage, the slide is keeping the output notation simple and general.

The main point is:

> the output is computed from the last hidden representation `h^{(L-1)}` using the parameters of the final layer.

Later in the course, the exact output form will be tied to the learning problem and loss function.

---

### Why Hidden Layers and Output Layer Are Separated

This separation is conceptually useful.

Hidden layers are usually thought of as:

- feature-transforming layers
- followed by nonlinear activations

The output layer is usually thought of as:

- the final prediction layer
- whose exact form depends on the prediction task

So the slide is already hinting that:

- internal representation learning happens in hidden layers,
- final prediction happens in the output layer.

That is why the notation splits them.

---

### The Diagram on the Right

The picture on the right shows three consecutive layers:

- `h^(l-1)`
- `h^(l)`
- `h^(l+1)`

with weight matrices:

- `\Theta^(l)`
- `\Theta^(l+1)`

This diagram is not about a specific number of neurons. It is illustrating the general pattern:

```text
previous layer -> current layer -> next layer
```

and each connection between adjacent layers has its own parameter matrix.

So the visual message is:

> a deep network is just repeated copies of the same layer-to-layer transformation.

That makes the network modular.

---

### Forward Pass: Evaluate Layer by Layer

The red statement on the slide says:

> **Forward pass:** evaluate layer by layer, `l = 1 -> L`

This is the operational meaning of the notation.

A **forward pass** means:

1. start with the input `h^(0) = x`
2. compute `h^(1)` from `h^(0)`
3. compute `h^(2)` from `h^(1)`
4. continue like this through all hidden layers
5. compute the final output at layer `L`

So the network is evaluated sequentially from left to right.

In compact pseudocode:

```text
h^(0) = x
for l = 1 to L-1:
    h^(l) = a(theta_0^(l) + Theta^(l) h^(l-1))
y_hat = theta_0^(L) + Theta^(L) h^(L-1)
```

That is the forward pass.

---

### Why the Forward Pass Matters

This may sound obvious, but it matters for two major reasons.

First, it tells us how prediction happens:

- input enters the network,
- each layer transforms it,
- the final layer produces `\hat{y}`.

Second, it prepares the ground for backpropagation:

- during the forward pass, intermediate values are computed layer by layer
- during backpropagation, error information will move backward through those same layers

So the forward pass is the computation that produces the prediction, and later the backward pass will explain how to learn from the prediction error.

---

### How Slide 5 Builds on Slides 3 and 4

Slide 3 said:

> stack more layers

Slide 4 said:

> one layer can be written compactly using matrix notation

Slide 5 now combines those two ideas:

> each indexed layer has its own matrix and bias, and the network is evaluated one layer at a time.

So this slide is where the notation becomes fully reusable for deep networks.

Instead of inventing new formulas for every architecture, we now have one generic layer rule.

That is a major simplification.

---

### A Useful Mental Model

A good way to think about the notation is:

- `h^(0)` = raw input
- `h^(1)` = first internal representation
- `h^(2)` = second internal representation
- ...
- `h^(L-1)` = last hidden representation
- `y_hat` = final prediction

So each `h^(l)` is the network's current "understanding" of the input after layer `l`.

This makes deep learning feel much less mysterious.

The network is not jumping directly from `x` to `y_hat`.
It is building the prediction gradually through a chain of internal representations.

---

### Why This Notation Is So Important Later

This slide may look like just bookkeeping, but it becomes essential for everything that comes next:

- defining the learning objective for the whole network
- writing loss functions in terms of `\hat{y}`
- computing gradients with respect to `\Theta^(l)` and `\theta_0^(l)`
- expressing backpropagation layer by layer

Without this notation, later derivations would become extremely cluttered.

So Slide 5 is not a side note. It is the language the rest of Session 3 will rely on.

---

### Main Takeaway

The whole slide can be summarized as:

> Layer notation lets us describe a deep network uniformly: `h^(0) = x`, each hidden layer computes `h^(l) = a(theta_0^(l) + Theta^(l) h^(l-1))`, and the forward pass evaluates these layers sequentially until the final output is produced.

This is the point where the network stops being "a collection of many neurons" and becomes "a sequence of indexed layers," which is the right viewpoint for deep learning.

---

## Slide 6: Deep Network as Function Composition

### What's On This Slide

This slide takes the layer notation from Slide 5 and expresses the whole deep network in a more abstract but very powerful way:

> a deep network is a **composition of functions**

This is one of the central ideas in deep learning.

The slide shows the network in three equivalent ways:

1. **Layer by layer**
2. **As a composed function**
3. **Written out as a nested formula**

Then it adds one more crucial idea:

4. **how many parameters the whole network contains**

So this slide is important because it connects:

- the architecture of the network,
- the mathematics of function composition,
- and the practical difficulty of training millions of parameters.

---

### View 1: Layer by Layer

The slide first writes the network as a sequence:

$$
x = h^{(0)} \xrightarrow{\Theta^{(1)}} h^{(1)} \xrightarrow{\Theta^{(2)}} \cdots \xrightarrow{\Theta^{(L)}} h^{(L)} = \hat{y}
$$

This is the most intuitive view.

It says:

- start with the input `x`,
- treat it as layer `0`,
- apply the first layer transformation to get `h^(1)`,
- apply the second layer transformation to get `h^(2)`,
- continue this process,
- and eventually get the final output `\hat{y}`.

So this is just the forward pass written in a compact visual way.

The important message is:

> each layer receives the representation from the previous layer and produces a new representation for the next one.

That is the sequential structure of a deep network.

---

### What This Means Conceptually

The layer-by-layer view tells us that a deep network does not jump straight from input to prediction.

Instead, it performs a chain of transformations:

```text
raw input
-> first representation
-> second representation
-> ...
-> final prediction
```

That means each layer is responsible for only part of the full computation.

This is one of the reasons deep networks are powerful:

- early layers can learn simple patterns,
- intermediate layers can combine them,
- later layers can turn them into task-specific predictions.

So depth gives the model a way to build complex behavior from repeated simpler steps.

---

### View 2: As a Composed Function

The slide then rewrites the same idea as:

$$
\hat{y} = f_{\theta}(x) = \left(h_{\theta}^{(L)} \circ \cdots \circ h_{\theta}^{(1)}\right)(x)
$$

This is the abstract mathematical version.

The symbol `\circ` means **function composition**.

If we write:

$$
(f \circ g)(x) = f(g(x))
$$

then "compose two functions" means:

- first apply `g` to `x`,
- then apply `f` to the result.

So when the slide writes:

$$
\left(h_{\theta}^{(L)} \circ \cdots \circ h_{\theta}^{(1)}\right)(x)
$$

it means:

- first apply layer 1 to `x`,
- then apply layer 2 to that result,
- then layer 3,
- and so on until layer `L`.

This is exactly the same computation as the forward pass. It is just written in function language.

---

### Why the Composition View Is So Important

This point is deeper than it may first appear.

A neural network is often introduced as:

- neurons,
- weights,
- layers,
- arrows in a diagram.

But mathematically, the full network is just a function:

$$
f_{\theta}: x \mapsto \hat{y}
$$

That is, it takes an input and produces an output.

The reason deep learning works the way it does is that this function is not built in one step. It is built as a **composition of many smaller functions**.

So the real mathematical identity of a deep network is:

> a parameterized function made by stacking simpler parameterized functions.

This viewpoint becomes very important in optimization and backpropagation.

---

### What `h_theta^(l)` Means Here

In the composition formula, the slide writes each layer as something like:

$$
h_{\theta}^{(l)}
$$

This means:

- the `l`th layer is being treated as a function,
- and that function depends on the parameters `\theta`.

So `h_\theta^(l)` is not just "the output at layer `l`."
It is the **mapping performed by layer `l`**.

In plain language:

- feed an input vector into layer `l`,
- use that layer's weights and biases,
- apply the required activation,
- return the new vector.

So the notation is shifting from:

- "what value does this layer produce?"

to:

- "what function does this layer implement?"

That is a very important conceptual upgrade.

---

### View 3: Written Out as a Nested Formula

The slide then expands the composition into a nested expression:

$$
\hat{y} =
g\left(
\theta_0^{(L)} + \Theta^{(L)}
\; a\left(
\cdots
a\left(
\theta_0^{(1)} + \Theta^{(1)} x
\right)
\cdots
\right)
\right)
$$

This is the fully written-out version of the network.

It looks complicated, but it is doing exactly what the earlier slides described:

1. Start from the input `x`
2. Apply the first affine transform and activation
3. Feed that result into the second layer
4. Repeat through all hidden layers
5. Apply the final output mapping

This formula makes the nesting explicit:

- the first layer's output sits inside the second layer,
- the second sits inside the third,
- and so on.

That is what composition really means in concrete terms.

---

### Why the Output Uses `g(.)` Here

A subtle but important detail on the slide is that the outermost function is written as `g(.)`, not `a(.)`.

That suggests:

- hidden layers use activation `a`
- the output layer may use a potentially different output function `g`

This is useful because different tasks need different output behaviors.

For example:

- regression may use a linear output,
- binary classification may use sigmoid,
- multiclass classification may use softmax.

So the slide is being slightly more general here than the earlier hidden-layer formulas.

The main message is:

> the inside of the network repeatedly applies hidden-layer transformations, and the final layer turns the last hidden representation into the prediction.

---

### From Nested Formula Back to Plain English

If the written-out formula feels intimidating, reduce it back to the story:

```text
Take x
-> transform it with layer 1
-> transform that result with layer 2
-> keep going
-> use the final layer to produce y_hat
```

That is all the formula is saying.

The notation looks dense because all layers are compressed into one expression, but conceptually it is still just repeated layer-by-layer computation.

---

### Total Number of Parameters

The slide then gives the total parameter count:

$$
|\theta| = \sum_{l=1}^{L} k_l \cdot (k_{l-1} + 1)
$$

This formula is extremely important.

It tells us how many learnable numbers the network contains altogether.

Let us decode the symbols:

- `k_l` = number of units in layer `l`
- `k_{l-1}` = number of units in the previous layer
- `k_0` = input dimension

For each layer `l`:

- there are `k_l * k_{l-1}` weights
- there are `k_l` biases

So total parameters in layer `l` are:

$$
k_l k_{l-1} + k_l = k_l (k_{l-1} + 1)
$$

Then the summation adds this quantity over all layers `l = 1, ..., L`.

That is how we count the parameters of the full network.

---

### Why the `+1` Appears

Students often wonder where the `+1` comes from in:

$$
k_l (k_{l-1} + 1)
$$

The reason is:

- `k_{l-1}` accounts for the incoming weights to each neuron
- the extra `1` accounts for the bias term of each neuron

So for each neuron in layer `l`, the parameter count is:

- one weight for each input coming from layer `l-1`
- plus one bias

Multiply that by the number of neurons `k_l`, and you get the formula for that layer.

---

### A Small Example of the Parameter Count

Suppose we have:

- input dimension `k_0 = 4`
- hidden layer 1 with `k_1 = 3`
- hidden layer 2 with `k_2 = 2`
- output layer with `k_3 = 1`

Then:

Layer 1 parameters:

$$
3(4 + 1) = 15
$$

Layer 2 parameters:

$$
2(3 + 1) = 8
$$

Layer 3 parameters:

$$
1(2 + 1) = 3
$$

Total:

$$
15 + 8 + 3 = 26
$$

So even a small deep network can already have many learnable parameters.

As networks grow wider and deeper, this number becomes huge.

---

### Why Parameter Count Matters

The slide ends with:

> **Real networks have millions - finding them is the challenge.**

This line is pointing directly toward the learning problem.

The issue is not just that deep networks are big.
The issue is that we need to find good values for all those parameters.

So once the number of parameters becomes very large:

- guessing them by hand is impossible,
- brute-force search is impossible,
- and training becomes a serious optimization problem.

That is why Session 3 must move next into:

- defining what "good parameters" means,
- measuring error with a loss function,
- and using gradient descent and backpropagation to search efficiently.

So the parameter-count formula is not just bookkeeping. It motivates the whole rest of the lecture.

---

### How Slide 6 Connects to the Previous Slides

Slide 3 said:

> a deep network is built by stacking layers

Slide 4 said:

> each layer can be written compactly with matrices

Slide 5 said:

> indexed layers are evaluated in a forward pass

Slide 6 now compresses all of that into one higher-level statement:

> the full deep network is a composition of layer functions with a very large set of parameters.

This is the point where the network starts looking less like a diagram and more like a parameterized mathematical object.

---

### Main Takeaway

The whole slide can be summarized as:

> A deep neural network is a function built by composing many layer functions, and the total number of learnable parameters grows as the sum of weights and biases across all layers.

That is why deep learning is both powerful and challenging:

- powerful because complex functions can be built step by step,
- challenging because the network may contain millions of parameters that must be learned from data.

---

## Slide 7: Dimensions

### What's On This Slide

This slide makes the notation from Slides 4, 5, and 6 concrete by showing an actual example network and working out the **dimensions** of its weight matrices.

The example network has:

- input dimension `d = 3`
- first hidden layer size `k_1 = 4`
- second hidden layer size `k_2 = 4`
- output dimension `m = 2`

So the architecture is:

```text
3 inputs -> 4 hidden units -> 4 hidden units -> 2 outputs
```

The slide then answers three practical questions:

1. What is the shape of each weight matrix?
2. What is the general rule for matrix dimensions?
3. How many total parameters does this network have?

This is a very important slide because it teaches you how to check whether your network equations are dimensionally correct.

---

### Why Dimensions Matter at All

In deep learning, writing the right formula is not enough.

You also have to make sure the shapes are compatible.

If the dimensions are wrong:

- matrix multiplication does not make sense,
- the layer cannot be computed,
- and in code you get shape mismatch errors.

So dimensions are not a minor detail.
They are part of understanding what the layer is actually doing.

This slide is training you to think:

> for every layer, how many values go in, and how many values come out?

Once you know that, the matrix shape follows naturally.

---

### The Example Network on the Slide

The slide gives a specific architecture:

- input vector with `d = 3` features
- first hidden layer with `k_1 = 4` neurons
- second hidden layer with `k_2 = 4` neurons
- output vector with `m = 2` values

So if we label the layers:

- layer 0: input, size `3`
- layer 1: hidden 1, size `4`
- layer 2: hidden 2, size `4`
- layer 3: output, size `2`

Then the forward flow is:

```text
x in R^3
-> h^(1) in R^4
-> h^(2) in R^4
-> y_hat in R^2
```

That immediately tells us the sizes of all intermediate vectors.

---

### Weight Matrix for the First Hidden Layer

The slide writes:

$$
\Theta^{(1)} \in \mathbb{R}^{4 \times 3}
$$

Why `4 x 3`?

Because:

- layer 1 has `4` neurons
- the previous layer has `3` values coming in

So each of the `4` neurons in hidden layer 1 needs:

- one weight for input 1,
- one weight for input 2,
- one weight for input 3

That means:

- `4` rows, one for each neuron in the current layer
- `3` columns, one for each value from the previous layer

So:

$$
\Theta^{(1)} \in \mathbb{R}^{k_1 \times d} = \mathbb{R}^{4 \times 3}
$$

This is the first example of the general dimension rule.

---

### Weight Matrix for the Second Hidden Layer

The slide then writes:

$$
\Theta^{(2)} \in \mathbb{R}^{4 \times 4}
$$

This time:

- hidden layer 2 has `4` neurons
- hidden layer 1 also has `4` neurons

So the current layer has `4` outputs, and the previous layer also provides `4` inputs.

That gives:

$$
\Theta^{(2)} \in \mathbb{R}^{k_2 \times k_1} = \mathbb{R}^{4 \times 4}
$$

This is a nice example because it shows that square matrices can appear, but that is only because the two adjacent layers happen to have the same size.

The fact that it is `4 x 4` is not because hidden layers are always square. It is only because this particular network has `k_1 = 4` and `k_2 = 4`.

---

### Weight Matrix for the Output Layer

Finally, the slide writes:

$$
\Theta^{(3)} \in \mathbb{R}^{2 \times 4}
$$

Why?

Because:

- the output layer has `m = 2` outputs
- the previous hidden layer has `k_2 = 4` units

So:

$$
\Theta^{(3)} \in \mathbb{R}^{m \times k_2} = \mathbb{R}^{2 \times 4}
$$

This means:

- there are `2` rows, one for each output unit
- there are `4` columns, one for each value coming from hidden layer 2

So each output unit looks at all 4 activations from the previous hidden layer.

---

### The General Rule

The slide summarizes the pattern as:

$$
\Theta^{(l)} \in \mathbb{R}^{k_l \times k_{l-1}}
$$

This is one of the most useful formulas in the whole notation section.

It says:

- `k_{l-1}` = size of the previous layer
- `k_l` = size of the current layer

Therefore:

- columns correspond to the previous layer
- rows correspond to the current layer

So if you know the sizes of two adjacent layers, you immediately know the shape of the weight matrix between them.

That is why the slide also states:

- **rows = neurons in layer `l`**
- **cols = neurons in layer `l - 1`**

This is the dimension rule you should memorize.

---

### Why Rows Correspond to the Current Layer

Each row of `\Theta^{(l)}` belongs to one neuron in layer `l`.

That row contains all incoming weights used by that neuron to combine the outputs from layer `l - 1`.

So if the current layer has `k_l` neurons, we need `k_l` rows.

This matches the idea from Slide 4:

> one row = all weights used by one neuron

The only difference now is that the notation is generalized to any layer `l`.

---

### Why Columns Correspond to the Previous Layer

Each column of `\Theta^{(l)}` corresponds to one neuron or feature from layer `l - 1`.

That column tells us how one value from the previous layer influences all neurons in the current layer.

So if layer `l - 1` has `k_{l-1}` units, the matrix needs `k_{l-1}` columns.

This gives the intuitive interpretation:

- rows answer: "what weights does each current neuron use?"
- columns answer: "where does each previous-layer value connect?"

This row-column interpretation is one of the best ways to avoid confusion.

---

### Checking the Vector Dimensions Too

The slide focuses on weight matrices, but it also helps to check the vector sizes.

In this example:

- `x \in \mathbb{R}^3`
- `h^(1) \in \mathbb{R}^4`
- `h^(2) \in \mathbb{R}^4`
- `\hat{y} \in \mathbb{R}^2`

Then the matrix multiplications make sense:

$$
\Theta^{(1)} x : (4 \times 3)(3 \times 1) = 4 \times 1
$$

$$
\Theta^{(2)} h^{(1)} : (4 \times 4)(4 \times 1) = 4 \times 1
$$

$$
\Theta^{(3)} h^{(2)} : (2 \times 4)(4 \times 1) = 2 \times 1
$$

So every step produces exactly the vector size we want for the next layer.

That is the real purpose of dimension reasoning.

---

### Total Number of Parameters

At the bottom, the slide computes:

$$
4(3 + 1) + 4(4 + 1) + 2(4 + 1) = 16 + 20 + 10 = 46
$$

This is the total number of learnable parameters in the example network.

Let us unpack it layer by layer.

For the first hidden layer:

- `4 x 3 = 12` weights
- `4` biases
- total = `16`

For the second hidden layer:

- `4 x 4 = 16` weights
- `4` biases
- total = `20`

For the output layer:

- `2 x 4 = 8` weights
- `2` biases
- total = `10`

Then:

$$
16 + 20 + 10 = 46
$$

So the network has 46 trainable numbers altogether.

---

### Why the Parameter Count Formula Matches Slide 6

In Slide 6, the general parameter formula was:

$$
|\theta| = \sum_{l=1}^{L} k_l (k_{l-1} + 1)
$$

Slide 7 is simply applying that formula to a specific architecture.

Here:

- `k_0 = 3`
- `k_1 = 4`
- `k_2 = 4`
- `k_3 = 2`

So:

$$
k_1(k_0 + 1) + k_2(k_1 + 1) + k_3(k_2 + 1)
$$

becomes:

$$
4(3 + 1) + 4(4 + 1) + 2(4 + 1)
$$

which is exactly the expression on the slide.

So Slide 7 is not introducing a new formula. It is showing you how to use the previous one correctly.

---

### What This Slide Is Really Teaching You

At a deeper level, the slide is teaching a habit:

> whenever you see a neural-network architecture, you should immediately be able to infer the shape of each weight matrix and the total number of parameters.

That skill is extremely useful because it helps with:

- understanding formulas,
- checking whether derivations make sense,
- implementing models correctly,
- debugging shape errors,
- and estimating model size.

So even though the slide looks computational, it is really building intuition for how deep networks are structured.

---

### Common Shortcut to Remember

A good shortcut is:

> **weight matrix shape = (current layer size) x (previous layer size)**

and

> **parameter count per layer = (current layer size) x ((previous layer size) + 1)**

The extra `+1` again comes from the bias term.

If you remember those two rules, you can reconstruct most of the slide very quickly.

---

### How Slide 7 Connects to the Bigger Picture

Slides 3 to 6 built the general language of deep networks:

- stacking layers
- matrix notation
- layer notation
- function composition
- parameter counting

Slide 7 now grounds that language in a specific example.

That matters because abstract notation can feel easy until you actually try to assign dimensions.

This slide shows that the formulas are not just symbolic. They correspond to concrete layer sizes and concrete parameter counts.

It is the bridge between theory and implementation.

---

### Main Takeaway

The whole slide can be summarized as:

> For a layer `l`, the weight matrix always has shape `k_l x k_{l-1}` because it maps outputs from the previous layer into neurons of the current layer, and the total number of parameters is found by adding weights and biases layer by layer.

So Slide 7 is teaching you how to read a network architecture numerically, not just visually.

---

## Slide 8: Batch Formulation

### What's On This Slide

This slide takes the layer equations you have seen so far and extends them from **one sample at a time** to **many samples at once**.

That is a very important shift, because in real deep-learning code we almost never process just one example in isolation.

Instead, we usually process a **batch** of `N` samples together.

So Slide 8 is answering the practical question:

> If one sample uses vector notation, how do we write the same layer computation for a whole batch of samples efficiently?

The slide gives the answer in the notation used by PyTorch, where the **batch dimension comes first**.

---

### Start with the Single-Sample Formula

The slide begins by reminding us of the single-sample hidden-layer computation:

$$
h^{(l)} = a\left(\theta_0^{(l)} + \Theta^{(l)} h^{(l-1)}\right) \in \mathbb{R}^{k_l}
$$

This is the layer rule we already know:

- `h^(l-1)` is the input to layer `l`
- `\Theta^(l)` is the weight matrix
- `\theta_0^(l)` is the bias vector
- `a(.)` is the activation function

For one sample, everything is a vector.

That is mathematically clean, but not how training is typically implemented at scale.

---

### Why We Want Batches

In practice, we usually have many training examples.

Instead of computing:

- one forward pass for sample 1,
- then one forward pass for sample 2,
- then one forward pass for sample 3,

we stack several samples together and process them in parallel.

This is useful because:

- it is computationally more efficient,
- modern hardware is optimized for matrix operations,
- and gradient-based training is usually done on mini-batches rather than single samples.

So Slide 8 is bridging the gap between clean math notation and actual deep-learning implementation.

---

### Batch of `N` Samples: Stack as Rows

The slide says:

$$
X \in \mathbb{R}^{N \times d}, \qquad H^{(l)} \in \mathbb{R}^{N \times k_l}
$$

This means:

- `N` = number of samples in the batch
- each row of `X` is one input example
- each row of `H^(l)` is the activation of one sample at layer `l`

So instead of:

- one input vector `x \in \mathbb{R}^d`

we now have:

- one input matrix `X \in \mathbb{R}^{N \times d}`

where the rows are:

```text
sample 1
sample 2
...
sample N
```

This is what "stack as rows" means.

---

### PyTorch Convention: Batch First

The slide explicitly says:

> **PyTorch convention: batch first**

This means the first dimension of the tensor is the batch size.

So if we have:

- `N` samples
- `d` input features

then PyTorch stores the input as:

```python
X.shape = (N, d)
```

and not `(d, N)`.

That choice affects how we write the matrix multiplication.

It is one of the most important practical conventions to internalize if you are working with PyTorch.

---

### The Batch Forward Pass Formula

The slide writes:

$$
H^{(l)} = a\left(\mathbf{1}\theta_0^{(l)\top} + H^{(l-1)}\left(\Theta^{(l)}\right)^{\top}\right)
$$

This is the batch version of the single-sample equation.

Let us unpack it carefully.

`H^(l-1)`:

- shape `N x k_{l-1}`
- one row per sample
- one column per feature coming from the previous layer

`\Theta^(l)^T`:

- shape `k_{l-1} x k_l`

So:

$$
H^{(l-1)}\left(\Theta^{(l)}\right)^{\top}
$$

has shape:

$$
(N \times k_{l-1})(k_{l-1} \times k_l) = N \times k_l
$$

which is exactly what we want for the output of layer `l` for all `N` samples.

---

### Why the Transpose Appears

This is one of the biggest points of confusion for students.

In the earlier mathematical notation, the weight matrix for one layer was written as:

$$
\Theta^{(l)} \in \mathbb{R}^{k_l \times k_{l-1}}
$$

That convention is natural when a single input is a column vector.

But in the batch setting on this slide:

- samples are stored as **rows**
- so the batch matrix is multiplied from the **left**

That means we need:

$$
H^{(l-1)} \cdot \left(\Theta^{(l)}\right)^\top
$$

instead of:

$$
\Theta^{(l)} H^{(l-1)}
$$

So the transpose appears because the storage convention changed from:

- single column-vector input

to:

- batch of row-wise samples

The computation is the same, but the matrix orientation changes to match the batch-first layout.

---

### What the Bias Term `1 theta_0^T` Means

The formula also includes:

$$
\mathbf{1}\theta_0^{(l)\top}
$$

This is just a compact way to say:

> copy the bias vector across all `N` rows in the batch

Here:

- `\mathbf{1}` is a column vector of ones with length `N`
- `\theta_0^{(l)\top}` is the row version of the bias vector

So their product creates an `N x k_l` matrix where every row is the same bias vector.

That way, the bias can be added to every sample in the batch at once.

In code, this is usually handled automatically by **broadcasting**.

So while the math writes `\mathbf{1}\theta_0^{\top}` explicitly, PyTorch often lets you just write `+ b`.

---

### Shape Check for the Bias Term

It helps to verify the dimensions:

- `\mathbf{1}` has shape `N x 1`
- `\theta_0^{(l)\top}` has shape `1 x k_l`

So:

$$
\mathbf{1}\theta_0^{(l)\top}
$$

has shape:

$$
(N \times 1)(1 \times k_l) = N \times k_l
$$

That matches the shape of:

$$
H^{(l-1)}\left(\Theta^{(l)}\right)^{\top}
$$

so the addition is valid.

Then the activation function `a(.)` is applied elementwise to the whole `N x k_l` matrix.

---

### What `H^(l)` Represents

The output:

$$
H^{(l)} \in \mathbb{R}^{N \times k_l}
$$

means:

- there are `N` rows, one for each sample in the batch
- there are `k_l` columns, one for each neuron in layer `l`

So entry `(n, j)` in `H^(l)` is:

> the activation of neuron `j` in layer `l` for sample `n`

That interpretation is very useful when you are debugging or reasoning about tensors.

---

### How This Connects to the Earlier Single-Sample Formula

It is important to see that nothing fundamentally new is happening.

For a single sample, we had:

$$
h^{(l)} = a\left(\theta_0^{(l)} + \Theta^{(l)} h^{(l-1)}\right)
$$

For a batch, we simply do that same computation for many samples at once.

So the batch formula is not a different neural network.
It is just the vectorized, implementation-friendly version of the same layer rule.

You can think of it as:

```text
single sample equation
applied to all rows simultaneously
```

That is the key idea.

---

### The Code on the Slide

The slide then gives the PyTorch-style summary:

```python
X.shape = (N, d)
W.shape = (k, d)
H = X @ W.T + b
```

This is exactly what PyTorch computes.

Let us decode it:

- `X` contains `N` samples, each with `d` features
- `W` contains `k` neurons, each with `d` incoming weights
- `W.T` has shape `(d, k)`
- `X @ W.T` therefore has shape `(N, k)`
- `b` is added to each row by broadcasting

So this code is the implementation form of the batch equation on the slide.

This is one of the most valuable parts of the slide because it connects lecture notation directly to real code.

---

### Why `W.shape = (k, d)` Matches the Earlier Math

Students sometimes notice that the code uses:

```python
W.shape = (k, d)
```

and wonder whether that contradicts the earlier notation.

It does not.

This is still the same idea:

- `k` rows = current layer neurons
- `d` columns = previous-layer features

That is exactly the same structure as:

$$
\Theta \in \mathbb{R}^{k \times d}
$$

The only reason we use `W.T` in the batch formula is that `X` is stored row-wise.

So the code and the math are consistent.

---

### Why Batch Formulation Matters for Training

This slide matters far beyond notation.

Training algorithms usually work with batches because:

- loss is often computed over multiple examples together
- gradients are usually estimated from batches
- GPUs and deep-learning libraries are optimized for large matrix operations

So if you want to understand modern deep-learning code, you must be comfortable moving from:

- single-sample vector formulas

to:

- batch matrix formulas

That is exactly the transition this slide is teaching.

---

### A Simple Way to Remember It

A good way to remember the batch rule is:

> single sample = vectors
> batch of samples = matrices whose rows are samples

So:

- `x` becomes `X`
- `h^(l)` becomes `H^(l)`
- bias addition becomes row-wise broadcasting
- the weight matrix gets transposed in the multiplication because the batch is row-oriented

If you keep that picture in mind, the slide becomes much easier to remember.

---

### How Slide 8 Connects to the Earlier Slides

The earlier slides built the mathematical description of a deep network:

- Slide 4: one layer in matrix notation
- Slide 5: indexed layers and forward pass
- Slide 6: the whole network as function composition
- Slide 7: dimensions of the matrices

Slide 8 now shows how this same mathematics is written in the form that actual deep-learning frameworks use.

So this slide is the bridge from:

- mathematical notation for one example

to:

- implementation-ready notation for many examples

That is why it is so important.

---

### Main Takeaway

The whole slide can be summarized as:

> In batch formulation, we stack samples as rows, replace vectors by matrices like `X` and `H^(l)`, and compute the whole layer for all samples at once using `H = X @ W.T + b` with the activation applied elementwise.

This is the practical form of the neural-network equations used in PyTorch and most modern deep-learning code.

---

## Slide 9: The Learning Problem

### What's On This Slide

This slide is a major conceptual transition in Session 3.

The earlier slides were mainly about:

- how to represent deep networks,
- how to write them mathematically,
- how to understand their dimensions,
- and how to compute a forward pass.

Slide 9 now asks the real training question:

> Once we have a network architecture, how do we make it actually learn?

That is why the slide is called **The Learning Problem**.

It is not talking about a single formula anymore.
It is defining the full objective of supervised learning.

---

### What We Have

The slide starts with two ingredients:

1. **A network `f_theta` with a forward pass**
2. **A dataset**

The dataset is written as:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

This notation means:

- we have `N` examples,
- each example has an input `x^(i)`,
- and a corresponding target or true output `y^(i)`.

So the dataset consists of input-output pairs.

For example:

- in image classification, `x^(i)` could be an image and `y^(i)` its label
- in house-price prediction, `x^(i)` could be house features and `y^(i)` the price
- in medical diagnosis, `x^(i)` could be patient measurements and `y^(i)` the diagnosis

So this slide is placing the network into the standard supervised-learning setup:

```text
input x -> model f_theta(x) -> prediction y_hat
compare prediction with true target y
```

---

### What `f_theta` Really Means

The notation `f_theta` is important.

It means:

- the network defines a function `f`
- but the exact function depends on the parameter values `theta`

So different choices of `theta` give different prediction functions.

This is exactly what the little graph on the right side is showing:

- `f_theta^a`
- `f_theta^b`
- `f_theta^c`

These are different candidate functions from the same model family.

In other words:

> the architecture defines the space of possible functions, and the parameter values decide which particular function we currently have.

That is a core learning-theory idea hidden in the slide.

---

### The Graph: Which `f_theta` Is Best?

The graph on the right shows data points and several possible curves.

This picture is asking a very important question:

> **Which `f_theta` is best?**

That is the whole learning problem in one sentence.

We are not asking:

- "Can the network compute something?"
- "Can we write the forward pass?"

We are asking:

- "Among all parameter settings, which one gives the best mapping from input to output?"

So the purpose of learning is to choose parameter values `theta` that make the network's function match the pattern in the data as well as possible.

---

### What We Want

The slide then states the goal:

> **Parameters `theta` such that `f_theta(x) ≈ y` on unseen data**

This sentence is extremely important.

It contains the real target of machine learning:

- not just fitting the examples we already have,
- but making good predictions on new examples we have not seen before.

That phrase **unseen data** is what makes the problem genuinely statistical.

The model should learn the underlying pattern, not merely memorize the training set.

So the true goal is:

```text
learn a function that generalizes
```

not just:

```text
learn a function that copies the training data
```

This is one of the most important ideas in the whole course.

---

### Why "Unseen Data" Matters So Much

Suppose a model performs perfectly on the training examples but fails badly on new inputs.

That model is not useful.

Why?

Because the whole reason we train a neural network is to use it on future data.

So when the slide says:

> `f_theta(x) ≈ y` on unseen data

it is emphasizing **generalization**.

Generalization means:

- the model captures a rule or pattern that extends beyond the examples it was shown

This is the real success criterion for machine learning.

So Slide 9 is not just about finding any good `theta`.
It is about finding `theta` that works beyond the training set.

---

### The Learning Problem in Plain Language

If we translate the slide into plain English, it says:

1. We have a model with many adjustable parameters
2. We have data telling us correct input-output behavior
3. We want to choose the parameters so the model predicts well
4. We especially want it to predict well on new data

That is the learning problem.

Everything else in the rest of Session 3 will be about solving that problem.

---

### Why the Slide Splits the Problem into Two Questions

The slide says there are **two open questions**:

1. How do we **measure** how wrong `f_theta` is?
2. How do we **find** good `theta` systematically?

This is a very clean decomposition.

Before we can train a network, we need both:

- a way to judge quality
- and a way to improve quality

If either one is missing, learning cannot really happen.

So the rest of the lecture naturally breaks into:

- **loss functions**
- **optimization**

These are the next major topics because they answer exactly these two questions.

---

### Open Question 1: How Do We Measure Wrongness?

The first question is:

> How do we measure how wrong `f_theta` is?

The slide answers:

> **loss function**

This is the next essential ingredient after defining the network.

A loss function gives us a number that tells us how bad the model's prediction is.

Without a loss function:

- we cannot compare two parameter settings precisely,
- we cannot say whether the model is improving,
- and we cannot define a clear learning objective.

So the loss function turns vague ideas like "good prediction" and "bad prediction" into a numerical quantity.

That is why it comes first.

---

### Open Question 2: How Do We Find Good `theta` Systematically?

The second question is:

> How do we find good `theta` systematically?

The slide answers:

> **optimization**

Even if we know how to measure error, that is still not enough.

We also need a procedure that searches through parameter space and improves the model.

That is what optimization does.

Optimization is the machinery that says:

- start from some initial parameter values
- measure performance
- adjust parameters in a direction that should improve performance
- repeat

So if the loss function tells us **how wrong** the model is, optimization tells us **how to reduce that wrongness**.

---

### Why These Two Questions Must Come in This Order

The ordering matters.

First, we need to define:

- what counts as good
- what counts as bad

Only after that can we ask:

- how do we improve the parameters?

So the logic is:

```text
Define the target numerically -> then search for parameters that achieve it
```

This is why the next slides will talk first about loss functions and then about optimization.

The lecture is following the exact structure introduced here.

---

### A Deeper Interpretation of the Slide

At a deeper level, Slide 9 says that a neural network alone is not enough.

A network architecture gives us:

- expressive power
- a parameterized family of functions

But learning requires more than expressiveness.

It requires:

- data,
- a criterion for success,
- and a method for searching parameter space.

So this slide marks the point where deep learning becomes an **optimization problem over functions defined by parameters**.

That is the real conceptual jump.

---

### How This Connects to the Earlier Slides

The earlier slides built the model side of the story:

- what the layers are
- how they compose
- how many parameters they have
- how the forward pass works

Slide 9 now adds the task side of the story:

- we have data
- we want good predictions
- we care about unseen data
- we need loss and optimization

So this is the slide where the session moves from:

- **describing the model**

to:

- **describing the learning objective**

That is why it is such an important turning point.

---

### A Good Short Summary of the Slide

You can compress the whole slide into this:

```text
We have a parameterized network and labeled data.
We want parameters that make the network predict correctly on new data.
To do that, we need:
1. a loss function
2. an optimization method
```

That is the entire roadmap of the learning problem.

---

### Main Takeaway

The whole slide can be summarized as:

> The learning problem is to choose parameter values `theta` so that the network `f_theta` maps inputs to correct outputs not only on the training set, but on unseen data, and solving that problem requires both a loss function and an optimization method.

This is the point where Session 3 stops asking "what is the network?" and starts asking "how do we make the network learn?"

---

## Slide 10: What Is a Loss Function?

### What's On This Slide

This slide answers the **first open question** from Slide 9:

> How do we measure how wrong `f_theta` is?

The answer is:

> **use a loss function**

This is a foundational idea in machine learning.

A neural network can produce predictions, but training cannot begin until we have a numerical way to say:

- how good a prediction is,
- how bad a prediction is,
- and whether one set of parameters is better than another.

That numerical rule is the **loss function**.

So Slide 10 is where the vague idea of "being wrong" becomes a precise scalar objective.

---

### The Goal: Measure Wrongness with One Scalar

The slide states the goal very clearly:

> **measure how wrong `f_theta` is with a single scalar**

This is important.

Why a **single scalar**?

Because optimization algorithms need one numerical quantity to minimize.

If the model makes a prediction, we need to compress the quality of that prediction into one number such that:

- smaller means better,
- larger means worse.

That way, learning becomes a problem of driving that number down.

So the loss function is the bridge between:

- prediction quality,
- and optimization.

Without that single scalar, the model would have no clear direction for improvement.

---

### Per-Sample Loss `\ell(\hat{y}, y)`

The slide first defines the **per-sample loss**, also called the **cost**:

$$
\ell(\hat{y}, y)
$$

This is a function that compares:

- `\hat{y}` = the model's prediction
- `y` = the true target

and returns a number telling us how bad that prediction is for one example.

So this is the local, sample-by-sample notion of error.

In plain language:

```text
prediction + truth -> one number measuring wrongness
```

This is the basic unit from which the full training objective is built.

---

### The Three Properties on the Slide

The slide lists three intuitive properties of a good per-sample loss:

1. `\ell(\hat{y}, y) = 0` when `\hat{y} = y`  
   That means a perfect prediction should incur no penalty.

2. `\ell(\hat{y}, y) > 0` otherwise  
   That means any mistake should produce some positive penalty.

3. Larger when more wrong  
   That means the loss should not only detect error, but also reflect **how much** error there is.

These three properties capture the basic behavior we want from a loss.

At a high level, a loss function should behave like a sensible score of mistake severity.

---

### Why "Larger When More Wrong" Matters

This third property is especially important.

Suppose the true value is `10`, and the model predicts:

- `9.9`
- `6`
- `-100`

All three predictions are technically "wrong."

But they are not equally wrong.

A good loss function should reflect that difference numerically.

That is why loss functions are more useful than simple right/wrong indicators.

They provide **graded feedback**.

And graded feedback is exactly what optimization needs.

---

### From One Sample to the Whole Dataset

A model is not trained on just one example.

It is trained on a dataset:

$$
\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}
$$

So once we know how to measure error on one sample, we need a way to measure error over **all training samples together**.

That is why the slide introduces the **dataset loss**:

$$
\mathcal{L}(\theta)
$$

This is the full training objective.

It tells us how good or bad the current parameter values `theta` are on the dataset as a whole.

---

### Dataset Loss `L(theta)`

The slide defines dataset loss as:

$$
\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \ell\left(f_{\theta}(x^{(i)}), y^{(i)}\right)
$$

This formula is one of the most important equations in the course.

Let us read it carefully.

For each training sample `i`:

1. take the input `x^(i)`
2. compute the model prediction `f_theta(x^(i))`
3. compare that prediction to the true target `y^(i)` using the per-sample loss `\ell`
4. get one scalar loss value

Then:

- add those losses over all `N` training samples
- divide by `N`

So the dataset loss is the **average per-sample loss over the training set**.

That is exactly what the slide says underneath the formula:

> **Average over all training samples**

---

### Why `L(theta)` Depends on `theta`

The notation `\mathcal{L}(\theta)` is important.

It means the total loss depends on the model parameters.

Why?

Because:

- the parameters `theta` determine the prediction `f_theta(x^(i))`
- the predictions determine the per-sample losses
- the average of those losses gives the dataset loss

So if we change `theta`, we change:

- the predictions,
- the losses,
- and therefore the overall value of `\mathcal{L}(\theta)`.

This is the quantity optimization will try to minimize.

---

### Why We Average Instead of Just Summing

The slide uses:

$$
\frac{1}{N}\sum_{i=1}^{N}
$$

instead of just the raw sum.

That is useful because averaging makes the loss scale less dependent on dataset size.

If one dataset has 100 examples and another has 10,000, the average loss is easier to interpret and compare than the raw sum.

So averaging gives us a normalized measure of how wrong the model is **per sample on average**.

In practice, both sum and average appear in different contexts, but the average is the standard conceptual choice for the training objective.

---

### Why the Loss Is the Model's Compass

The slide ends with the phrase:

> **Loss: model compass - what is "better"?**

That is an excellent way to think about it.

A compass does not magically move you to the destination.
It tells you which direction is the right one.

The loss function plays the same role:

- it does not update the parameters by itself,
- but it tells us whether one parameter setting is better or worse than another.

So the loss defines the meaning of improvement.

Without a loss function, optimization would have no notion of direction.

That is why the loss comes before gradient descent in the lecture.

---

### Choice of `\ell` Depends on the Task

The slide emphasizes:

> **Choice of `\ell` depends on the task**

This is extremely important.

There is no single universal loss function that is best for every problem.

Different tasks need different ways of measuring prediction error.

The slide gives three standard examples:

- **Regression** -> **Mean Squared Error**
- **Binary classification** -> **Binary Cross-Entropy**
- **Multiclass** -> **Cross-Entropy**

So the loss function is not arbitrary.
It must match the structure of the prediction task.

---

### Regression: Mean Squared Error

For regression, the model predicts a continuous value, like:

- house price
- temperature
- sales
- distance

In this setting, we care about **numerical closeness** between prediction and target.

That is why **Mean Squared Error (MSE)** is a natural choice.

At a high level, MSE:

- looks at the difference between prediction and truth,
- squares it,
- and averages across examples.

This makes larger numerical mistakes incur larger penalties.

So regression losses are about measuring how far the predicted number is from the true number.

---

### Binary Classification: Binary Cross-Entropy

In binary classification, the output is one of two classes, such as:

- spam vs not spam
- fracture vs no fracture
- fraud vs not fraud

Here the model often predicts a probability-like value for the positive class.

So we want a loss that strongly rewards confident correct predictions and strongly penalizes confident wrong predictions.

That is why **Binary Cross-Entropy** is commonly used.

This loss is designed for two-class probabilistic outputs.

So unlike regression, the key issue is not "how far apart are two numbers?" but rather:

> how well does the predicted probability align with the true class?

---

### Multiclass Classification: Cross-Entropy

In multiclass classification, there are more than two possible classes, such as:

- cat, dog, horse
- digits 0 to 9
- disease A, disease B, disease C

Now the model outputs a score or probability distribution over many classes.

In this setting, **Cross-Entropy** is the standard loss.

It rewards the model when it assigns high probability to the correct class and penalizes it when the correct class receives low probability.

So again, the loss is chosen to match the structure of the task.

This is why the slide puts the task-loss pairs in a table.

---

### Why the Loss Function Is Not Just a Technical Detail

Students sometimes treat the loss function as just one more formula to memorize.

That is a mistake.

The loss function is one of the most important modeling choices in machine learning because it defines:

- what counts as error,
- how severely different mistakes are penalized,
- and therefore what kind of behavior the model will learn.

So choosing the loss is part of defining the learning problem itself.

The architecture tells us what the model **can represent**.
The loss tells us what the model is being asked to **care about**.

That is a very deep distinction.

---

### How This Connects to Slide 9

Slide 9 ended with two open questions:

1. How do we measure how wrong `f_theta` is?
2. How do we find good `theta` systematically?

Slide 10 answers the first one.

It says:

- we define a per-sample loss `\ell(\hat{y}, y)`
- we average it over the dataset to get `\mathcal{L}(\theta)`
- and we use that scalar as the criterion of model quality

So this slide is the formal beginning of the optimization story.

Before we can minimize anything, we first need to know **what** we are minimizing.

That is exactly what the loss function provides.

---

### A Good Short Summary of the Slide

You can compress the whole slide into this:

```text
A loss function turns prediction error into a number.
For one example, that number is ell(y_hat, y).
For the whole training set, we average those numbers to get L(theta).
The choice of loss depends on the task.
```

That is the core message.

---

### Main Takeaway

The whole slide can be summarized as:

> A loss function is the scalar measure of prediction error that tells us how wrong the model is, and the dataset loss `L(theta)` is the average of that error over all training samples, providing the objective that optimization will later try to minimize.

This is the slide where "wrong prediction" becomes a mathematically usable training objective.

---

## Slide 11: Regression: Mean Squared Error

### What's On This Slide

This slide specializes the general loss-function idea from Slide 10 to the case of **regression**.

In regression, the target is a real-valued number, so the model is not choosing among classes. It is trying to predict a continuous quantity such as:

- a price,
- a temperature,
- a distance,
- a score,
- or a measurement.

So Slide 11 answers the question:

> If the task is regression, what should the output layer look like, and what loss should we use?

The slide’s answer is:

- use a **raw scalar output**
- and measure error using **Mean Squared Error (MSE)**

---

### Target and Prediction Dimensions Must Match

The slide begins with:

> **Target & prediction dim match**

This is a simple but crucial point.

If the true target `y` is a scalar, then the model prediction `\hat{y}` must also be a scalar.

The slide writes:

$$
y \in \mathbb{R}, \qquad \hat{y} \in \mathbb{R}
$$

So in this setup:

- the target is one real number,
- the prediction is one real number,
- and the loss compares those two numbers directly.

This is why regression feels more like "predict the correct number" than "choose the correct class."

The output object and the target object must live in the same space, otherwise the error comparison does not make sense.

---

### Why This Slide Uses a Scalar Output

The slide is showing the simplest regression case:

> one input example -> one scalar prediction

That is why it focuses on:

$$
y \in \mathbb{R}, \qquad \hat{y} \in \mathbb{R}
$$

In more advanced settings, regression can also have vector outputs, but the core idea is the same:

- the prediction and target must have matching dimensions,
- and the loss is applied componentwise or aggregated appropriately.

For this slide, the professor keeps it simple by presenting the single-output version first.

---

### Last Layer Has No Activation

The slide then says:

> **Last layer has no activation**

and writes the raw output as:

$$
\hat{y} = h^{(L)} = \theta_0^{(L)} + \Theta^{(L)} h^{(L-1)}
$$

This means the final layer is just a linear or affine transformation:

- take the last hidden representation `h^(L-1)`
- multiply by the final-layer weights
- add the final-layer bias
- output the result directly

There is no sigmoid, no softmax, and no extra nonlinear squashing at the end in this setup.

---

### Why No Activation in the Last Layer?

This choice makes sense for regression because the target is a real number.

If we put a restrictive activation on the final layer, we would artificially constrain the possible outputs.

For example:

- sigmoid would force the output into `(0, 1)`
- `tanh` would force the output into `(-1, 1)`
- ReLU would force the output to be nonnegative

But many regression targets can be:

- negative or positive,
- small or large,
- not confined to a narrow interval.

So a raw linear output is the most natural default because it lets the network predict any real value.

That is why this slide pairs regression with **no final activation**.

---

### The Per-Sample Mean Squared Error

The slide defines the per-sample loss as:

$$
\ell(\hat{y}, y) = (\hat{y} - y)^2
$$

This means:

1. compute the prediction error `\hat{y} - y`
2. square it

That gives a nonnegative number measuring how wrong the prediction is for one sample.

If the prediction is perfect, then:

$$
\hat{y} = y \quad \Rightarrow \quad (\hat{y} - y)^2 = 0
$$

So the loss is zero when the prediction is exact.

If the prediction is wrong, the squared difference becomes positive.

---

### Why We Square the Error

The square is doing important work.

First, it removes the sign.

If we used just `\hat{y} - y`, then:

- positive errors and negative errors could cancel each other out

That would be a bad error measure.

Squaring prevents cancellation, because both:

- `(+3)^2 = 9`
- `(-3)^2 = 9`

So underprediction and overprediction are both penalized.

Second, squaring makes larger mistakes count much more heavily.

For example:

- error `1` gives squared error `1`
- error `2` gives squared error `4`
- error `5` gives squared error `25`

So MSE punishes big mistakes strongly.

That is exactly why the slide says:

> **Large errors quadratically**

meaning the penalty grows with the square of the error magnitude.

---

### Dataset Loss for Regression

The slide then defines the dataset loss:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \left(\hat{y}^{(i)} - y^{(i)}\right)^2
$$

This is just the average of the squared errors over all training samples.

Equivalently, since `\hat{y}^{(i)} = f_\theta(x^{(i)})`, we can also write:

$$
\mathcal{L}(\theta) = \frac{1}{N}\sum_{i=1}^{N} \left(f_\theta(x^{(i)}) - y^{(i)}\right)^2
$$

So for every sample `i`:

1. the network predicts `\hat{y}^{(i)}`
2. we compare it to the true scalar `y^{(i)}`
3. we square the difference
4. we average over the dataset

That gives us one scalar objective telling us how well the model is doing on the training set.

---

### Why This Is Called "Mean Squared Error"

The name comes directly from the formula:

- **Error** = `\hat{y} - y`
- **Squared** = we square that error
- **Mean** = we average across samples

So the name is not arbitrary. It describes exactly what the loss does.

This is useful because many machine-learning terms sound abstract until you unpack them. MSE is actually one of the most literal names in the field.

---

### Interpreting the Plot on the Right

The figure on the right shows:

- data points,
- a regression curve `f_\theta`,
- and an example vertical error between the predicted value and the true point.

That vertical gap is the regression error for one sample:

$$
\hat{y}^{(i)} - y^{(i)}
$$

MSE takes that gap and squares it.

So visually, the slide is saying:

> for regression, training tries to make the model's predicted curve pass close to the observed data points, and the loss measures the size of those deviations.

This is a very intuitive picture of what regression learning means.

---

### Intuition Bullet 1: `+ / -` Values -> Absolute or Square

The first intuition bullet on the slide says:

> **+ / - values -> absolute or square**

This is reminding you that raw prediction errors can be positive or negative:

- if `\hat{y} > y`, the error is positive
- if `\hat{y} < y`, the error is negative

But we do not want positive and negative errors to cancel out.

So we transform the raw error into a nonnegative quantity.

Two common ways to do that are:

- absolute value
- square

This slide chooses the square, which leads to MSE.

---

### Intuition Bullet 2: Large Errors Quadratically

The second bullet says:

> **Large errors quadratically**

This is emphasizing the main personality of MSE:

- small errors matter,
- but large errors matter much more.

That can be good when we really want to discourage big misses.

For example, if predicting `100` instead of `10` would be a serious failure, MSE will penalize that very strongly.

So MSE is not just measuring error. It is saying:

> large deviations are especially undesirable.

That preference is built into the square.

---

### Intuition Bullet 3: Differentiable

The slide’s third bullet says:

> **Differentiable**

This matters because Session 3 is building toward gradient descent and backpropagation.

Optimization methods like gradient descent need derivatives.

MSE is smooth and differentiable with respect to the model output, which makes it easy to optimize using gradient-based methods.

That is one of the major reasons MSE is such a standard regression loss.

It is not only intuitive, but also mathematically convenient for learning.

---

### Why MSE Is a Natural Fit for Regression

Putting all of this together, MSE fits regression well because:

- regression targets are numeric,
- the prediction error is numeric,
- squaring gives a clean nonnegative penalty,
- large mistakes are penalized strongly,
- and the result is differentiable for optimization.

So MSE is both conceptually natural and computationally practical.

That is why it is often the default loss for basic regression models.

---

### How This Connects to Slide 10

Slide 10 introduced the general idea:

- per-sample loss `\ell(\hat{y}, y)`
- dataset loss `\mathcal{L}(\theta)`
- different tasks use different losses

Slide 11 now instantiates that idea for one specific task:

- **task**: regression
- **output type**: scalar real number
- **loss**: Mean Squared Error

So this slide is the first concrete example of how the abstract loss-function framework is applied in practice.

---

### A Good Short Summary of the Slide

You can compress the whole slide into this:

```text
For scalar regression, the model outputs a raw real number.
The loss for one example is (y_hat - y)^2.
The training objective is the average of these squared errors over the dataset.
```

That is the full core idea.

---

### Main Takeaway

The whole slide can be summarized as:

> In scalar regression, the network typically uses a raw linear output with no final activation, and training measures prediction quality using Mean Squared Error, which averages squared prediction gaps and penalizes large numerical mistakes strongly.

This is the standard starting point for regression in neural networks.

---

## Slide 12: The Optimization Problem

### What's On This Slide

This slide answers the **second open question** from Slide 9:

> How do we find good `theta` systematically?

Slide 10 defined the loss.
Slide 11 gave a concrete example of a loss for regression.

Now Slide 12 says:

> once the loss is defined, learning becomes an **optimization problem**

That is the central idea of modern deep learning.

We are no longer just describing a network or measuring its error.
We are now asking how to choose parameter values that make that error as small as possible.

---

### The Goal: Minimize the Loss

The slide states the goal clearly:

> **find parameters that minimize loss**

This means we want parameter values `theta` that make the objective function `\mathcal{L}(\theta)` as small as possible.

So training can now be phrased very compactly:

```text
Pick the parameters that give the smallest loss
```

This is the formal mathematical version of:

```text
find the network that makes the best predictions
```

because "best predictions" have now been translated into "lowest loss."

---

### The Meaning of

$$
\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)
$$

This is the key equation on the slide.

It looks compact, but it carries a lot of meaning.

Let us unpack it:

- `\mathcal{L}(\theta)` = the loss as a function of the parameters
- `\min` = we want the smallest possible value
- `\arg\min` = we want the **argument**, meaning the parameter values, that achieve that smallest value
- `\theta^*` = the optimal parameter setting

So this equation does **not** mean:

> "the smallest loss value itself"

It means:

> "the parameter vector `theta` that makes the loss as small as possible"

That distinction is important.

If we wrote:

$$
\min_{\theta} \mathcal{L}(\theta)
$$

that would refer to the minimum loss value.

But:

$$
\arg\min_{\theta} \mathcal{L}(\theta)
$$

refers to the parameter values where that minimum occurs.

---

### Why Training Is an Optimization Problem

This slide is saying that once we define the loss, training becomes a search problem in parameter space.

We have:

- a huge set of possible parameter values
- and for each parameter setting, the loss tells us how good or bad it is

So we can imagine a landscape:

- some points in parameter space give high loss
- some give lower loss
- and we want to move toward parameter settings with smaller loss

That is what optimization means here.

Deep learning is therefore not just about neural networks.
It is about optimizing a large, complicated loss function defined by those networks.

---

### Why Not Solve It Analytically?

The slide then asks a very natural question:

> **Why not solve analytically?**

That is, why do we not just derive the exact optimal `theta` in one clean formula?

In simple problems from basic algebra or linear regression, closed-form solutions sometimes exist.

But in deep learning, that usually does not happen.

The slide gives three reasons.

These reasons are fundamental, and each one matters.

---

### Reason 1: `L(theta)` Is Nonlinear in `theta`

The first reason is:

> **`\mathcal{L}(\theta)` is nonlinear in `theta` - no closed form**

This comes directly from the structure of a deep network.

A neural network contains:

- matrix multiplications,
- nonlinear activation functions,
- compositions of many layers,
- and often complicated output mappings and loss functions.

So the final loss as a function of the parameters is highly nonlinear.

That means we usually cannot rearrange the equations and solve for the best parameters directly in a simple formula.

In other words:

> the mathematics is too intertwined to produce a neat exact solution.

That is what "no closed form" means here.

---

### What "No Closed Form" Means

A **closed-form solution** is a direct formula you can write down for the optimum.

For example, in some simpler settings, you can derive an answer explicitly without doing iterative search.

Deep networks generally do not give us that luxury.

Instead of:

```text
theta* = some exact formula
```

we usually only have:

```text
start somewhere
improve theta step by step
```

So the absence of a closed-form solution is one of the main reasons training is an algorithmic process rather than a one-step computation.

---

### Reason 2: `theta` Has Millions of Dimensions in Practice

The second reason is:

> **`theta` has millions of dimensions in practice**

This is a scale problem.

In a small toy example, the parameter vector may have just a few dozen numbers.

But real neural networks often have:

- millions,
- tens of millions,
- or even billions of parameters.

That means the optimization is happening in an extremely high-dimensional space.

So even if the loss were easier to describe, directly searching such a huge space would still be difficult.

This is why parameter count matters so much.

The network is not just learning one number or ten numbers.
It is coordinating enormous numbers of interacting parameters at once.

---

### Why High Dimensionality Makes Training Hard

When the parameter space is huge:

- we cannot exhaustively search all possibilities
- we cannot visualize the full landscape easily
- and each parameter can affect many others indirectly through the network

So optimization in deep learning is challenging not only because the loss is nonlinear, but also because the search space is enormous.

This is one reason why efficient optimization methods are essential.

Naive search would be hopelessly impractical.

---

### Reason 3: `L(theta)` Is Non-Convex

The third reason is:

> **`\mathcal{L}(\theta)` is non-convex - many local minima**

This is another major difficulty.

A **convex** optimization problem has a very nice property:

- any local minimum is also a global minimum

That makes optimization much easier conceptually.

But deep-network losses are typically **non-convex**, which means the loss landscape can contain:

- many valleys,
- many hills,
- flat regions,
- saddle points,
- and many local minima.

So the surface is much more complicated than a simple bowl shape.

That is why training a deep network is not just a neat clean optimization exercise.
It is a messy high-dimensional search over a difficult landscape.

---

### What "Local Minima" Means

A **local minimum** is a point where the loss is lower than nearby points, but not necessarily lower than every point in the whole space.

So it is "locally good" but not guaranteed to be globally best.

This matters because an optimization algorithm may descend into one of these valleys and stop improving significantly there, even though a better valley might exist elsewhere.

So the slide is highlighting that deep-learning optimization is not just about moving downhill.
It is about moving downhill on a very complicated terrain.

---

### The Strategy: Iterative, Gradient-Based Optimization

After listing the difficulties, the slide gives the practical answer:

> **Strategy: iterative, gradient-based optimization**

This is the key conclusion.

Since we cannot solve for `theta^*` directly, we instead:

1. start from an initial parameter setting
2. compute how the loss changes with respect to the parameters
3. update the parameters a little
4. repeat many times

That is what "iterative" means:

- we improve the parameters step by step rather than all at once

And that is what "gradient-based" means:

- we use gradient information to decide which direction should reduce the loss

This is exactly the doorway into gradient descent and backpropagation.

---

### Why Gradients Are the Right Tool Here

In a huge nonlinear space, we need local information that tells us how the loss responds to small parameter changes.

That is what gradients provide.

A gradient tells us:

- which direction increases the loss
- and therefore which opposite direction should reduce it

So rather than blindly guessing parameter updates, gradient-based methods use the geometry of the loss surface locally.

That makes learning systematic rather than random.

This is why the course is about to move from the abstract optimization problem to the specific method of gradient descent.

---

### A Useful Mental Picture

It helps to imagine training as hiking in a foggy mountain landscape.

- the height of the ground is the loss
- your position is the current parameter vector `theta`
- you want to get to a low valley

But:

- the landscape is huge
- it is nonlinear
- it has many valleys
- and you cannot see the whole map at once

So instead of solving the whole problem globally in one shot, you repeatedly look at the local slope and take a step downhill.

That is the intuition behind iterative gradient-based optimization.

---

### How Slide 12 Connects to the Earlier Slides

The flow of the lecture is now very clear:

- Slides 3 to 8: define the network and its notation
- Slide 9: define the learning problem
- Slide 10: define the loss
- Slide 11: show one concrete loss for regression
- Slide 12: formulate training as minimizing that loss

So Slide 12 is the exact point where deep learning becomes an optimization problem in the formal mathematical sense.

From here, the lecture can naturally continue to:

- gradient descent
- backpropagation
- parameter updates

because the optimization target has now been clearly stated.

---

### A Good Short Summary of the Slide

You can compress the whole slide into this:

```text
Training means choosing parameters that minimize the loss.
We write that as theta* = argmin_theta L(theta).
We cannot solve this directly because the loss is nonlinear, high-dimensional, and non-convex.
So we use iterative gradient-based optimization.
```

That is the full roadmap of the slide.

---

### Main Takeaway

The whole slide can be summarized as:

> Once the loss is defined, learning becomes the problem of finding parameter values that minimize that loss, and because deep-network losses are nonlinear, high-dimensional, and non-convex, we solve this using iterative gradient-based optimization rather than an exact analytical formula.

This is the mathematical starting point of gradient descent.

---

## Slide 13: Gradient Descent

### What's On This Slide

This slide introduces the first concrete optimization method used to train neural networks:

> **gradient descent**

Slide 12 said that training is an optimization problem and that the strategy will be **iterative, gradient-based optimization**.

Slide 13 now explains exactly what that means.

The slide gives four core ideas:

1. what the **gradient** is
2. what the **gradient descent update rule** is
3. why we move **opposite** to the gradient
4. what the **learning rate** does

So this slide is the first place where the optimization story becomes algorithmic.

---

### The Gradient `∇_theta L(theta)`

The slide starts with:

$$
\nabla_{\theta}\mathcal{L}(\theta)
$$

This symbol means the **gradient of the loss with respect to the parameters**.

At a high level, the gradient tells us:

- how sensitive the loss is to each parameter
- and in which direction the loss increases most steeply

So the gradient is a mathematical summary of how the loss changes locally around the current parameter setting.

If `theta` has many components, then the gradient is a vector with the same number of components.

That is why the slide also says:

> **One partial derivative per parameter**

So if:

$$
\theta = (\theta_1, \theta_2, \dots, \theta_p)
$$

then:

$$
\nabla_{\theta}\mathcal{L}(\theta)
=
\left(
\frac{\partial \mathcal{L}}{\partial \theta_1},
\frac{\partial \mathcal{L}}{\partial \theta_2},
\dots,
\frac{\partial \mathcal{L}}{\partial \theta_p}
\right)
$$

Each component tells us how the loss changes when we change one parameter slightly.

---

### Why the Gradient Points in the Direction of Steepest Ascent

The slide says:

> **Points in direction of steepest ascent**

This is one of the most important facts about gradients.

It means:

- if you move a tiny amount in the direction of the gradient,
- the loss increases as quickly as possible locally.

So the gradient points **uphill** on the loss surface.

That is why, if our goal is to **minimize** the loss, we do not move in the direction of the gradient.

We move in the **opposite** direction.

That opposite direction is the local steepest descent direction.

This is the core geometric idea behind gradient descent.

---

### What a Partial Derivative Means Here

The slide highlights one example partial derivative:

$$
\frac{\partial \mathcal{L}}{\partial \theta_j}
$$

This measures:

> how much the loss changes if we nudge parameter `theta_j` slightly, while holding the others fixed

So:

- a large positive value means increasing `theta_j` increases the loss strongly
- a large negative value means increasing `theta_j` decreases the loss
- a value near zero means the loss is not very sensitive to that parameter locally

When we collect all these partial derivatives together, we get the full gradient.

So the gradient is essentially the local sensitivity report for all parameters at once.

---

### The Gradient Descent Update Rule

The central equation on the slide is:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}(\theta)
$$

This is the gradient descent update.

It means:

1. compute the gradient at the current parameter setting
2. scale it by a positive number `eta`
3. subtract that quantity from the current parameters
4. use the result as the new parameter vector

So we update the parameters by taking a small step in the downhill direction.

That is the entire core algorithm in one line.

---

### Why We Subtract the Gradient

The slide says:

> **Step opposite to gradient -> go downhill**

This follows directly from the fact that the gradient points toward steepest ascent.

If:

- the gradient points uphill

then:

- the negative gradient points downhill

So the update

$$
\theta - \eta \nabla_{\theta}\mathcal{L}(\theta)
$$

means:

> move a little bit in the direction that should reduce the loss

If we added the gradient instead of subtracting it, we would be doing **gradient ascent**, which would increase the loss.

That is the opposite of what we want during training.

---

### The Role of the Learning Rate `eta`

The slide says:

> **`eta > 0`: learning rate - controls step size**

The learning rate `eta` tells us how big each update step should be.

This is crucial.

The gradient tells us the direction, but the learning rate tells us how far to move in that direction.

So:

- the gradient = direction
- the learning rate = step size

Together, they define the update.

This is why the slide labels the arrow on the plot as something like:

$$
\eta \cdot \nabla \mathcal{L}
$$

because the step length depends on both the gradient and `eta`.

---

### What Happens If the Learning Rate Is Too Small

If `eta` is too small:

- each update barely changes the parameters
- the loss may decrease, but very slowly
- training can take a long time

So a tiny learning rate is usually safe but inefficient.

The model moves downhill in very small steps and may need many iterations to make meaningful progress.

---

### What Happens If the Learning Rate Is Too Large

If `eta` is too large:

- updates can overshoot the valley
- the loss can bounce around
- optimization can become unstable
- training may even diverge instead of improving

So a large learning rate can make training fast if it is well chosen, but harmful if it is too aggressive.

This is why the learning rate is one of the most important hyperparameters in deep learning.

The slide does not go into tuning strategies yet, but it correctly emphasizes that `eta` controls the step size.

---

### Interpreting the Plot on the Right

The figure on the right gives a 1-dimensional picture of the loss landscape.

The horizontal axis is:

$$
\theta
$$

and the vertical axis is:

$$
\mathcal{L}(\theta)
$$

The curve is shaped like a bowl, and the point `\theta^*` at the bottom is the minimizer.

The points labeled `\theta_0`, `\theta_1`, `\theta_2` show successive iterations of gradient descent.

The picture is illustrating:

- start somewhere on the slope
- compute the local gradient
- step downhill
- repeat
- gradually approach the minimum

So the visual message matches the update equation exactly.

---

### Why the Plot Is Shown in One Dimension

Real neural networks do not have one parameter.
They have many, often millions.

So the true loss surface lives in a huge high-dimensional space.

The slide uses a 1D picture because it is easier to visualize the basic idea:

- slope
- uphill
- downhill
- minimum

So the picture is not the full reality of deep learning optimization.
It is a simplified illustration of the local logic of the update rule.

The exact same idea extends to many dimensions, where the slope becomes the gradient vector.

---

### Repeat Until Convergence

The slide’s final bullet says:

> **Repeat until convergence**

This means gradient descent is not a one-step method.

We keep applying the update again and again:

1. compute the gradient
2. update parameters
3. compute the new gradient
4. update again

until the parameters or the loss stop changing much, or until some stopping condition is reached.

In plain language:

```text
look at the slope
take a downhill step
repeat many times
```

That repeated stepping is what gradually trains the model.

---

### What "Convergence" Means

Convergence usually means that the optimization has reached a point where:

- the loss is no longer decreasing much
- the gradient is small
- or parameter updates have become very small

It does **not** necessarily mean we found the absolute global minimum.

In deep learning, convergence usually means:

> we have reached a stable or good-enough solution for training

This is important because the loss surface may be complicated and non-convex, as Slide 12 explained.

So convergence is usually about practical stabilization, not perfect mathematical certainty.

---

### A More Operational View of the Algorithm

You can think of gradient descent as this loop:

```text
initialize theta
repeat:
    compute loss L(theta)
    compute gradient ∇theta L(theta)
    update theta <- theta - eta ∇theta L(theta)
```

This is the simplest training loop at the heart of many neural-network methods.

Of course, real training often uses batches, more advanced optimizers, and backpropagation to compute gradients efficiently, but conceptually this is the foundation.

---

### Why This Slide Matters for Backpropagation

Gradient descent tells us **how to use** gradients.

But it does not yet tell us **how to compute** those gradients efficiently for a deep network with many layers.

That is exactly where backpropagation comes in.

So the relationship is:

- **loss** defines what we want to minimize
- **gradient descent** defines how we update parameters using gradient information
- **backpropagation** computes that gradient information efficiently

This slide is the middle piece of that chain.

---

### How Slide 13 Connects to Slide 12

Slide 12 said:

> use iterative, gradient-based optimization

Slide 13 now specifies the simplest such method:

> gradient descent

So Slide 13 is the direct operational answer to the optimization problem formulated on Slide 12.

It turns the abstract objective

$$
\theta^* = \arg\min_{\theta}\mathcal{L}(\theta)
$$

into a concrete step-by-step algorithm for trying to approach that optimum.

---

### A Good Short Summary of the Slide

You can compress the whole slide into this:

```text
The gradient tells us which direction increases the loss fastest.
So to reduce the loss, we move in the opposite direction.
The update is theta <- theta - eta ∇theta L(theta).
The learning rate eta controls how big each step is.
Repeat this until the optimization stabilizes.
```

That is the full core idea.

---

### Main Takeaway

The whole slide can be summarized as:

> Gradient descent minimizes the loss by repeatedly updating the parameters in the direction opposite to the gradient, using the learning rate to control step size, so that the model gradually moves downhill on the loss surface toward a minimum.

This is the basic optimization rule behind neural-network training.

---

## Slide 14: The Key Question

### What's On This Slide

This is the closing conceptual slide of Session 3.

It does not introduce a new loss or a new optimization rule. Instead, it asks the most important practical question that remains after Slide 13:

> If gradient descent needs the gradient, how do we actually compute that gradient for a deep network?

That is why the slide is titled:

> **The Key Question**

Everything in the session has been building toward this moment.

The earlier slides established:

- what a deep network is,
- how to write it as layered compositions,
- what the learning problem is,
- what a loss function is,
- and how gradient descent uses gradients.

Now Slide 14 asks the missing implementation question:

> how do we differentiate efficiently through a network that is made of many composed layers?

The slide’s answer is:

> **Computation graphs + Backpropagation**

---

### Gradient Descent Needs the Gradient

The slide starts with:

> **Gradient descent needs:**

and then writes:

$$
\nabla_{\theta}\mathcal{L}(\theta) = \frac{\partial \mathcal{L}}{\partial \theta}
$$

The exact notation is compact, but the idea is simple:

gradient descent cannot update the parameters unless it knows how the loss changes with respect to those parameters.

So to train the model, we need quantities like:

- `∂L / ∂theta_1`
- `∂L / ∂theta_2`
- `∂L / ∂theta_3`
- and so on for all parameters

That is what the full gradient represents.

Slide 13 already told us how to **use** this gradient:

$$
\theta \leftarrow \theta - \eta \nabla_{\theta}\mathcal{L}(\theta)
$$

But that update rule only helps once the gradient is actually available.

So Slide 14 is pointing out the hidden computational bottleneck:

> where does `\nabla_\theta \mathcal{L}(\theta)` come from?

---

### Why This Is a Nontrivial Problem

At first glance, someone might think:

> "Just differentiate the loss."

But the problem is that a deep network is not a single simple expression.

It is a composition of many operations:

- matrix multiplications,
- bias additions,
- activation functions,
- layer after layer,
- and finally the loss function itself.

So the loss depends on the parameters only indirectly through a long chain of intermediate computations.

That is why the slide says:

> **`f_theta` is a composition of many layers**

This is the real difficulty.

The deeper the network becomes, the longer this dependency chain becomes.

So the question is not merely:

> can we differentiate?

It is:

> can we differentiate **efficiently** through a long chain of composed functions?

That word efficiently is the whole point of the slide.

---

### Why Layer Composition Makes Differentiation Hard

Suppose a network looks like:

```text
x -> h^(1) -> h^(2) -> h^(3) -> ... -> y_hat -> loss
```

Then an early parameter, such as a weight in layer 1, affects the loss only through everything that happens after it.

That means if we want:

$$
\frac{\partial \mathcal{L}}{\partial \theta^{(1)}}
$$

we must account for how changing that parameter affects:

- layer 1 output,
- which affects layer 2 output,
- which affects layer 3 output,
- and so on,
- until it eventually changes the loss.

So each parameter influences the loss through a chain of dependencies.

That is exactly the kind of situation where the **chain rule** from calculus becomes essential.

Slide 14 is essentially setting up the question:

> how do we apply the chain rule systematically across a whole deep network?

---

### The Hidden Role of the Chain Rule

The slide does not explicitly write the chain rule, but it is the mathematics behind everything that follows.

When a function is built by composition, like:

$$
\mathcal{L}(\theta) = \ell(f_{\theta}(x), y)
$$

and `f_theta` itself is composed of many layer functions, then differentiating the loss requires repeatedly applying the chain rule.

That means:

- compute how the loss changes with respect to the output,
- then how the output changes with respect to the previous layer,
- then how that previous layer changes with respect to an earlier layer,
- and continue backward through the network.

So the gradient computation problem is really:

> how do we organize all those chain-rule calculations without doing massive redundant work?

That is the exact motivation for backpropagation.

---

### Why Naive Differentiation Would Be Too Expensive

In principle, you could try to compute derivatives separately for each parameter one by one.

But for a deep network, that would be extremely inefficient because:

- there may be millions of parameters,
- many derivative calculations reuse the same intermediate quantities,
- and recomputing everything from scratch for each parameter would waste enormous amounts of work.

So the challenge is not only correctness.
It is computational efficiency.

We need a method that exploits the layered structure of the network so that shared intermediate derivatives are reused rather than recomputed.

That is exactly what backpropagation does.

---

### Why the Slide Says "Computation Graphs"

The slide’s answer is not just:

> backpropagation

It says:

> **Computation graphs + Backpropagation**

This is important because backpropagation works naturally when the forward computation is represented as a **computation graph**.

A computation graph is a structured way to represent:

- inputs,
- intermediate variables,
- operations,
- and outputs

as nodes and edges in a graph.

In plain language, it is a map of how the final output was computed step by step.

That structure makes it possible to differentiate systematically by moving backward through the graph.

So the graph gives us the organization, and backpropagation gives us the backward derivative procedure.

---

### What a Computation Graph Really Gives Us

If we write the forward pass as a graph, then every quantity in the network is connected to the quantities it depends on.

For example:

- one node might represent `z = Wx + b`
- another node might represent `h = a(z)`
- another node might represent the loss `L(h, y)`

Now the full network is no longer just a big messy formula.
It is a chain of elementary operations.

That is powerful because differentiating a complicated expression becomes much easier when we break it into small local pieces.

So a computation graph turns:

- one huge derivative problem

into:

- many small derivative problems connected together

This is the right perspective for deep learning.

---

### What Backpropagation Does

Backpropagation is the algorithm that computes gradients efficiently by moving **backward** through the network after the forward pass.

At a high level, it does this:

1. run the forward pass and compute the loss
2. start from the loss
3. propagate derivative information backward through each operation
4. accumulate gradients for each parameter

So backpropagation is not a separate learning objective and not a replacement for gradient descent.

It is the mechanism that tells gradient descent what the gradient actually is.

This relationship is crucial:

- **loss** defines what we want to minimize
- **gradient descent** defines how parameters are updated
- **backpropagation** computes the gradients used in those updates

Slide 14 is explicitly pointing at that final piece.

---

### Why Backpropagation Is Efficient

The word **efficiently** on the slide matters a lot.

Backpropagation is efficient because it reuses intermediate results.

During the forward pass, the network already computes many intermediate quantities, such as:

- linear combinations,
- activations,
- outputs of each layer

During the backward pass, backpropagation reuses that structure and applies local derivatives step by step.

So instead of independently differentiating the full network with respect to each parameter from scratch, it shares computations across parameters.

That is why it scales to deep networks with many layers.

Without this efficiency, modern neural-network training would be far less practical.

---

### Why This Is the Real Bridge to the Next Session Topic

Conceptually, Slide 14 is a bridge slide.

It says:

- we know what the gradient descent update looks like
- but we still do not know how to obtain the gradient for a deep model
- so the next topic must explain efficient differentiation

This is the natural transition into:

- computation graphs
- chain rule in network form
- backpropagation equations

So Slide 14 is not teaching the full backprop algorithm yet.
It is preparing the motivation for it.

That is why the slide is short, but very important.

---

### How Slide 14 Connects to the Whole Session

The full session arc now becomes clear:

- Slides 3 to 8: define deep networks and notation
- Slide 9: define the learning problem
- Slides 10 and 11: define loss functions
- Slide 12: formulate optimization
- Slide 13: introduce gradient descent
- Slide 14: ask how to compute gradients efficiently

So Slide 14 is the final missing link in the training story.

Without it, the session would still be incomplete, because we would know:

- what to minimize
- and how to update parameters once gradients are known

but not:

- how to actually obtain those gradients in a deep network

That is exactly the gap this slide identifies.

---

### A Good Short Summary of the Slide

You can compress the whole slide into this:

```text
Gradient descent needs gradients of the loss with respect to all parameters.
But a deep network is a composition of many layers, so those derivatives are not trivial to compute efficiently.
That is why we use computation graphs and backpropagation.
```

That is the whole message.

---

### Main Takeaway

The whole slide can be summarized as:

> The key remaining challenge in training deep networks is not just knowing that gradient descent needs `∇_theta L(theta)`, but computing that gradient efficiently through many composed layers, and the standard solution is to represent the forward computation as a computation graph and use backpropagation to propagate derivatives backward through it.

This is the slide that turns the optimization story into the backpropagation story.
