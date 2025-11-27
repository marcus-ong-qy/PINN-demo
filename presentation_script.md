# PINN Demo Presentation Script (4 minutes)

Follow this script while running cells in `pinn.ipynb`

## Demo Length Assessment

Your demo is well-suited for 4 minutes! Here's why:

✅ Strengths:

1. Clear narrative arc: Problem → Baseline → Physics → Boundary Conditions → Applications
2. Three distinct visual comparisons: The plots show dramatic differences between approaches
3. Natural pauses: Training epochs give you breathing room and let the audience process
4. Good balance: ~2:50 for live demo + ~1:10 for context and applications

📊 Content Breakdown:

- Introduction & Setup (30s): Sets context for why PINNs matter
- Data Generation (40s): Shows ground truth and noisy data
- Regular NN (20s): Quick baseline failure
- PINN with Physics (40s): Main concept explained in detail
- PINN with BC (40s): Completing the solution
- Summary & Applications (70s): Connects to real-world use cases

💡 Tips for Delivery:

- The expanded script includes more explanation of why and how, not just what
- I added pauses where you should let plots speak for themselves
- The real-world applications section at the end (2:50-4:00) is crucial - it answers "why should I care?"
- If you need to adjust timing, the applications section is easiest to expand/contract

Your demo has just the right amount of content - not too simple, not overwhelming. The visual progression from overfitting → correct shape with wrong position → perfect result is very pedagogical!

---

## [0:00-0:15] Introduction & Setup

**[Run import cell and seed cell]**

"Alright! Now let's see PINNs in action with a live demo.

I'm going to show you what happens when we try to learn a simple free-fall trajectory from noisy data - first with a regular neural network, then with a PINN. The problem is simple: a ball thrown upward under gravity, governed by d²y/dt² = -g.

where $y$ is the position of the object, $t$ is time, and $g$ is the acceleration due to gravity (approximately 9.81 m/s²).

Basically, we are modelling a parabola.

We will try three different approaches: (read off notebook)

---

## [0:15-0:30] Generate Ground Truth Data

**[Run cell: `generate_data()` and plot]**

"Here's our ground truth: a perfect parabola with initial velocity of 9.81 m/s - exactly equal to g - which means the ball lands perfectly at ground level after 2 seconds."

**[Show the smooth parabola plot - brief pause]**

"But of course, we never have perfect data in practice."

---

## [0:50-1:10] Add Noise to Simulate Real-World Data

**[Run cell: `sample_data()`]**

"In reality, we don't have perfect measurements. Maybe we're using sensors with noise, or there are environmental factors we can't control. So let me simulate real-world conditions.

First, I'm sampling just 20 points from this trajectory - sparse data, like what you might have in an actual experiment."

**[Run cells: `add_noise()` and `add_random_offsets()`]**

"Then I'm adding Gaussian noise to every point, plus some large random measurement errors to a few points - simulating outliers or sensor glitches."

**[Show the noisy scattered points - pause]**

"Look at how messy this is! This is what we have to work with. Can a neural network learn the correct physics from this?"

**[Run the dataframe and tensor preparation cells without commentary]**

---

## [1:10-1:30] Regular Neural Network Fails

**[Run cells: create `Regular_NN`, train it]**

"Let's start with a baseline: a standard neural network with no physics knowledge. This is just a simple feed-forward network - one input for time, two hidden layers of 20 neurons each with Tanh activations, and one output for position."

**[Let it train, showing epochs - you can comment on loss decreasing]**

"The loss is going down, so it's learning something..."

**[Run evaluation and plot Regular_NN results]**

"But look at this result! The network completely overfits the noise. It learned this wiggly, high-order polynomial-like curve that tries to hit every noisy point, including the outliers.

This is mathematically fitting the data, but it's physically nonsense. A ball under gravity cannot move like this. The network has no concept of physics."

---

## [1:30-2:10] PINN with Physics Loss

"Now let's add physics. This is where PINNs become powerful."

**[Scroll to show the physics_loss explanation markdown]**

"From Newton's second law, we know that the acceleration of a falling object is constant: d²y/dt² = -g. This is a differential equation that must be satisfied at every point in time.

Here's how I encode this into the neural network: I use PyTorch's automatic differentiation to compute the first derivative dy/dt, then the second derivative d²y/dt² of the network's output. Then I compute the physics residual (which is derived from the acceleration equation): d²y/dt² + g. If the network is respecting physics, this residual should be zero everywhere.

So I minimize the squared residual as part of my loss function." (That is, the square of the residual $\left(\frac{d^2y}{dt^2} + g\right)^2$)

The error term will be added to the overall loss function, scaled by a `lambda` value, similar to a regularisation term.

`loss = data_loss + lambda * physics_loss`

**[Run cells: create PINN, train with `lambda_phys=1`, `lambda_bc=0`]**

"Now the total loss has two terms: **Data Loss** to fit the noisy points, plus **Physics Loss** to satisfy the differential equation. I'm using lambda_phys equals 1 to balance them."

**[Let it train - watch the loss components being printed]**

"Notice how the training now shows both data loss and physics loss being minimized together."

**[Run evaluation and plot PINN without BC]**

"And look at the result! We get a smooth parabolic curve. The physics constraint prevented it from overfitting.

But obviously, this is wrong! Look at the boundaries: the ball starts underground and doesn't land at ground level. In summary, the shape is right, but the position is wrong."

---

## [2:10-2:50] Add Boundary Conditions

"The problem is we haven't told the network WHERE the ball should be at specific times. We've only told it HOW it should accelerate. That's where boundary conditions come in."

**[Scroll to show the bc_loss explanation markdown]**

"In physics problems, boundary conditions specify the state at the edges of your domain. For our problem, we know the ball starts and ends at ground level: y(0) = 0 and y(2) = 0.

I encode this by adding another loss term: the squared difference between the predicted position and the expected position at these boundary times. So we minimize: (y(0) - 0)² + (y(2) - 0)²."

**[Run cells: create PINN with BC, train with `lambda_phys=1`, `lambda_bc=10`]**

"Now our total loss has three components: **Data Loss** to fit the measurements, **Physics Loss** to satisfy the differential equation, and **Boundary Condition Loss** to pin down the start and end positions.

I'm using lambda_bc equals 10 to give the boundary conditions strong weight."

**[Let it train - watch all three loss components]**

"See how all three losses are being balanced during training."

**[Run evaluation and plot PINN with BC]**

"And there we have it! A perfect parabola that starts at ground level, follows the correct physics of gravity throughout its trajectory, and lands exactly where it should - all learned from that messy, noisy data.

This is the power of PINNs: by incorporating physics as constraints, we get results that are both data-driven and physically consistent."

---

## [2:50-4:00] Summary & Real-World Applications

**[Scroll to summary markdown and show it]**

"Let me quickly summarize what we've learned:"

_(read off from notebook)_

<!-- **[Gesture to the three main results if you can scroll through them]**

"We trained three models on the same noisy data:

1. A regular neural network that overfitted and produced unphysical results
2. A PINN with physics loss that learned the correct parabolic shape but wrong boundaries
3. A PINN with both physics and boundary conditions that produced a perfect, physically accurate trajectory

The key insight is that the physics loss acts like intelligent regularization. Instead of just penalizing model complexity, we're penalizing violations of fundamental physical laws.

**[Scroll to the final markdown about real-world applications]**

Now, you might think: 'This is just a parabola - we already know the answer!' True, but this was for demonstration. The real power of PINNs comes when you have complex physics problems where:

- Data is expensive or sparse - like oceanographic measurements or medical imaging
- The governing equations are known but analytical solutions are impossible - like turbulent fluid flow
- You need to interpolate or extrapolate beyond your measurements while respecting physics

PINNs are being used today for modeling blood flow in cardiovascular systems, predicting material behavior under stress, modeling atmospheric dynamics, and solving complex partial differential equations that would be computationally expensive with traditional methods.

The beautiful thing is: PINNs bridge two worlds. They have the flexibility of neural networks to handle noisy, incomplete data, but they're constrained by the reliability of physical laws.

Thank you!" -->

---

## Timing Tips

- **Total time:** 4 minutes
- The script above is fairly comprehensive - adjust pacing as needed
- If running under time: expand on real-world applications at the end
- If running over time: shorten the boundary conditions explanation or skip reading some markdown cells
- Let the training epochs run naturally - this gives natural pauses
- The visual comparison of the three plots is your strongest moment - make sure to emphasize it

---

## Key Takeaway Points

1. **Regular NN** overfits noisy data → wiggly, unphysical curve
2. **PINN with physics loss** → correct parabola shape, wrong boundaries
3. **PINN with physics + BC loss** → perfect physically-accurate result

**The "aha moment":** Physics acts like smart regularization that guides the network!

