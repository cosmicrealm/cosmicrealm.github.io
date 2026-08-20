---
layout: home
permalink: /
title: "Jinyang Zhang"
seo_title: "Jinyang Zhang - Generative AI Researcher"
excerpt: "Generative AI researcher focused on digital humans, image restoration, multimodal generation, and efficient model inference."
author_profile: false
---

<section class="home-hero">
  <p class="home-hero__eyebrow">Jinyang Zhang</p>
  <h1>Generative AI researcher.</h1>
  <p class="home-hero__lead">I focus on digital humans, image restoration, multimodal generation, and efficient model inference. I am currently seeking full-time research and engineering opportunities.</p>
  <div class="hero-actions">
    <a href="mailto:{{ site.author.email }}">Email</a>
    <a href="https://github.com/{{ site.author.github }}" target="_blank" rel="noopener">GitHub</a>
    <a href="{{ '/cv/' | relative_url }}">CV</a>
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Selected Work</h2>
    <a class="section-heading__link" href="{{ '/projects/' | relative_url }}">All projects</a>
  </div>
  {% assign featured_home_projects = site.data.projects | where: "homepage", true | sort: "date" | reverse %}
  <div class="home-card-grid">
    {% for project in featured_home_projects limit:3 %}
      {% include home-project-card.html project=project %}
    {% endfor %}
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Representative Publications</h2>
    <a class="section-heading__link" href="{{ '/publications/' | relative_url }}">All publications</a>
  </div>
  {% assign selected_publications = site.publications | sort: "date" | reverse %}
  <div class="publication-grid">
    {% for publication in selected_publications limit:3 %}
      {% include publication-card.html publication=publication %}
    {% endfor %}
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Foundations</h2>
    <a class="section-heading__link" href="{{ '/foundations/' | relative_url }}">All foundations</a>
  </div>
  {% assign featured_foundations = site.data.foundations | where: "featured", true | sort: "date" | reverse %}
  <div class="home-card-grid">
    {% for foundation in featured_foundations limit:3 %}
      {% include home-foundation-card.html foundation=foundation %}
    {% endfor %}
  </div>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Recent Writing</h2>
    <a class="section-heading__link" href="{{ '/writing/' | relative_url }}">All writing</a>
  </div>
  <ul class="compact-list compact-list--dated">
    {% for post in site.posts limit:4 %}
      <li>
        <time datetime="{{ post.date | date_to_xmlschema }}">{{ post.date | date: "%Y.%m.%d" }}</time>
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </li>
    {% endfor %}
  </ul>
</section>
