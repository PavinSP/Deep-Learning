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
