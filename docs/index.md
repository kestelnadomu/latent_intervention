# Counterfactual Latent Representations

<!-- Motivation -->
Sensitive attributes such as gender \cite{intro_gender} and race \cite{intro_race} continue to influence screening and hiring decisions, which is why such tasks are regulated under EU anti-discrimination law \cite{weerts2023}.
While algorithmic hiring offers an opportunity to reduce human stereotyping and discrimination, algorithms often inherit societal biases from their training data. As a result, fairness concerns in algorithmic hiring are actively discussed by a growing body of research \cite{fabris_fairness_2025}.

<!-- Counterfactual Fairness Perspective -->
From a counterfactual fairness perspective, the central question in this dilemma is: ``How would a person's CV look like if its sensitive attribute(s) were different? Would this person be treated differently?''
Answering this question requires causal reasoning to generate this person's data in a counterfactual world, which is a non-trivial task, particularly with unstructured high-dimensional data, such as text.

<!-- Contribution -->
In this work, we propose a methodology to generate a counterfactual latent representation of a text input. We use a pre-trained variational autoencoder (VAE) to map text to a latent representation, and train a neural manipulator model to transform these representations analogously to a causal intervention (abduction-action-prediction) on observable data w.r.t. a sensitive attribute. This manipulator is supervised by a semantic decoder that maps a fraction of the latent space to a structured representation, thus providing a consistency signal during training. We sketch the architecture, propose a training objective for the manipulator, and discuss its limitations (see Figure~\ref{fig:demo}).

Our work adds to the emerging research at the intersection of neurosymbolic AI and algorithmic fairness \cite[e.g.,][]{wagner_neural_2021, adriaensen_problog4fairness_2026, heilmann2026neurosymbolic} and illustrates the potential of neurosymbolic approaches for flexible bias mitigation.
<!-- 
\begin{figure}
    \centering
    \begin{subfigure}{0.48\linewidth}
        \centering
        \includegraphics[width=.8\linewidth]{img/demo.png}
        \caption{General architecture demonstration: an encoder $f$ encodes text data as a representation $Z$ in ``latent space'' (shaded). A decoder $g$ extracts some relevant structured features $S$ from $Z$. We define a symbolic manipulation function $h_S$ that transforms $S$ to $S'$, e.g., by calculating the counterfactual. Knowing $S'$, we use $g$ as a consistency bridge to train an analogue manipulation function $h_Z$ in latent space.}
        \label{fig:demo}  
        \Description{A chart visualizing the data representations $X,Z,S$, the functions $f,g,h_S,h_Z$ as entities and their process direction as arrows.}
    \end{subfigure}    
    \hfill
    \begin{subfigure}{0.48\linewidth}
        \centering
        \includegraphics[width=\linewidth]{img/worlds.png}
        \caption{Graphical illustration of the procedure:
        %Phase 0) A pre-trained encoder $f$ (black) maps the observable data $X$ to its latent representation $Z$.
        Phase 1) A semantic decoder $g$ (orange) is trained to retrieve some structured features $S$ from $Z$.
        Phase 2) A counterfactual function $h_S$ (blue) is derived from an SCM to transform $S$ to $S'$. An analogue latent function $h_Z$ (red) is trained to transform $Z$ to $Z'$, such that $g(Z') = S'$.
        A prediction model $p$ (pink) then predicts a target variable $Y$ (or $Y'$) from $Z$ (or $Z'$).}
        \label{fig:analogy}
        \Description{A chart visualizing the data representations $X,Z,S$ and the outcome $Y$ in the factual - and the counterfactual world. $X$ leads to $Z$ with a black arrow. $Z$ leads to $Z'$ with a red arrow. $Z$ and $Z'$ lead to $S$ or $S'$ with orange arrows. $S$ leads to $S'$ with a blue arrow. $Z$ and $Z'$ lead to $Y$ or $Y'$ with pink arrows.}
    \end{subfigure}
    %\caption{Two perspectives on the proposed method: a) a general view on the manipulation function $h$; b) the specific procedure.}
\end{figure} -->
