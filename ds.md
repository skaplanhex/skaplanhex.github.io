---
layout: page
title: Data Science
permalink: /datascience/
---

<!-- Here are some projects I've been working on:

[Titanic Dataset Analysis (Kaggle)](https://github.com/skaplanhex/skaplanhex.github.io/blob/master/notebooks/titanic/Titanic.ipynb)

More to be added soon! -->

{% for post in site.posts %}
  <!-- {{ post.categories }} -->
  {% if post.categories contains "datascience" %}
  {{ post.date | date_to_string }}
  <h2>
  <a href="{{ post.url }}">
    {{ post.title }}
  </a>
  </h2>
  <br>
  {{ post.excerpt }}
  {% endif %}
{% endfor %}