---
permalink: /
title: "Jinyang Zhang"
seo_title: "Jinyang Zhang - Projects, Publications, Blogs, and CV"
excerpt: "Personal website for projects, publications, blogs, and CV."
author_profile: true
redirect_from:
  - /about/
  - /about.html
---

<section class="home-section home-section--about">
  <h2>About Me</h2>
  <p>
    I am an algorithm engineer focused on AIGC systems that connect research models with deployable workflows, with experience across speech-driven facial generation, digital human synthesis, face restoration, ComfyUI restoration workflows, and practical tools for model development and daily execution. I am currently seeking new opportunities where I can continue building reliable generative AI systems from research prototypes to usable products.
  </p>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Projects</h2>
    <a class="section-heading__link" href="{{ '/projects/' | relative_url }}">All projects</a>
  </div>
  {% assign featured_projects = site.data.projects | where: "featured", true | sort: "date" | reverse %}
  <ul class="compact-list project-compact-list">
    {% for project in featured_projects limit:4 %}
      {% assign primary_link = project.links | first %}
      <li>
        {% if project.teaser %}
          {% if primary_link.url contains "http" %}
            <a class="list-thumb list-thumb--project" href="{{ primary_link.url }}" target="_blank" rel="noopener" aria-label="{{ project.name }}">
              <img src="{{ project.teaser | relative_url }}" alt="{{ project.teaser_alt | default: project.name }}">
            </a>
          {% else %}
            <a class="list-thumb list-thumb--project" href="{{ primary_link.url | relative_url }}" aria-label="{{ project.name }}">
              <img src="{{ project.teaser | relative_url }}" alt="{{ project.teaser_alt | default: project.name }}">
            </a>
          {% endif %}
        {% endif %}
        <div class="project-compact-list__body">
          <div class="project-compact-list__main">
            {% if primary_link.url contains "http" %}
              <a href="{{ primary_link.url }}" target="_blank" rel="noopener">{{ project.name }}</a>
            {% else %}
              <a href="{{ primary_link.url | relative_url }}">{{ project.name }}</a>
            {% endif %}
            <span>{{ project.display_date }} · {{ project.highlight }}</span>
          </div>
          <p>{{ project.summary }}</p>
        </div>
      </li>
    {% endfor %}
  </ul>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Publications</h2>
    <a class="section-heading__link" href="{{ '/publications/' | relative_url }}">All publications</a>
  </div>
  {% assign selected_publications = site.publications | sort: "date" | reverse %}
  <ol class="publication-list">
    {% for publication in selected_publications limit:4 %}
      <li>
        {% if publication.teaser %}
          <a class="list-thumb list-thumb--publication" href="{{ publication.url | relative_url }}" aria-label="{{ publication.title }}">
            <img src="{{ publication.teaser | relative_url }}" alt="{{ publication.teaser_alt | default: publication.title }}">
          </a>
        {% endif %}
        <div class="publication-list__main">
          <a href="{{ publication.url | relative_url }}">{{ publication.title }}</a>
          <span>{{ publication.venue }} · {{ publication.date | date: "%Y" }}</span>
          {% if publication.summary %}
            <p>{{ publication.summary }}</p>
          {% endif %}
          {% if publication.projecturl or publication.codeurl %}
            <div class="publication-list__links" aria-label="{{ publication.title }} links">
              {% if publication.projecturl %}
                <a href="{{ publication.projecturl }}" target="_blank" rel="noopener">Project</a>
              {% endif %}
              {% if publication.codeurl %}
                <a href="{{ publication.codeurl }}" target="_blank" rel="noopener">{{ publication.codelabel | default: "Code" }}</a>
              {% endif %}
            </div>
          {% endif %}
        </div>
      </li>
    {% endfor %}
  </ol>
</section>

<section class="home-section">
  <div class="section-heading section-heading--line">
    <h2>Blogs</h2>
    <a class="section-heading__link" href="{{ '/year-archive/' | relative_url }}">All blogs</a>
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
