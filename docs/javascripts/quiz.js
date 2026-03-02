/* Quiz toggle behavior: click Show answer to reveal, click again to hide. */
(function () {
  "use strict";

  let answerCounter = 0;

  function appendInlineCodeText(container, text) {
    const raw = String(text);
    const parts = raw.split(/(`[^`]+`)/g);
    for (const part of parts) {
      if (!part) {
        continue;
      }
      if (part.startsWith("`") && part.endsWith("`") && part.length >= 2) {
        const code = document.createElement("code");
        code.textContent = part.slice(1, -1);
        container.appendChild(code);
        continue;
      }
      container.appendChild(document.createTextNode(part));
    }
  }

  function enhanceQuizItems(root) {
    const quizNodes = root.querySelectorAll(".cm-quiz");
    for (const node of quizNodes) {
      if (!(node instanceof HTMLElement)) {
        continue;
      }
      if (node.dataset.enhanced === "true") {
        continue;
      }
      const answerText = node.dataset.answer;
      if (!answerText) {
        continue;
      }

      node.dataset.enhanced = "true";
      node.removeAttribute("tabindex");
      const questionText = (node.textContent || "").trim();
      node.replaceChildren();

      const question = document.createElement("span");
      question.className = "cm-quiz-question";
      question.textContent = questionText;

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "cm-quiz-toggle";
      toggle.textContent = "Show answer";
      toggle.setAttribute("aria-expanded", "false");

      const answer = document.createElement("div");
      answer.className = "cm-quiz-answer";
      answer.hidden = true;
      answer.id = "cm-quiz-answer-" + answerCounter++;
      toggle.setAttribute("aria-controls", answer.id);

      const answerPrefix = document.createElement("strong");
      answerPrefix.textContent = "Answer:";
      answer.appendChild(answerPrefix);
      answer.appendChild(document.createTextNode(" "));
      appendInlineCodeText(answer, answerText);

      toggle.addEventListener("click", function () {
        const shouldShow = answer.hidden;
        answer.hidden = !shouldShow;
        toggle.textContent = shouldShow ? "Hide answer" : "Show answer";
        toggle.setAttribute("aria-expanded", String(shouldShow));
      });

      node.appendChild(question);
      node.appendChild(document.createTextNode(" "));
      node.appendChild(toggle);
      node.appendChild(answer);
    }
  }

  function boot() {
    enhanceQuizItems(document);
  }

  if (typeof document$ !== "undefined" && document$.subscribe) {
    document$.subscribe(boot);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
