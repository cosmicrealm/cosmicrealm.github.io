---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

<ul class="index-list project-index-list">
{% for project in site.data.projects %}
  <li>
    {% assign primary_link = project.links | first %}
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
    <div class="project-index-list__body">
      {% if primary_link.url contains "http" %}
        <a class="project-index-list__title" href="{{ primary_link.url }}" target="_blank" rel="noopener">{{ project.name }}</a>
      {% else %}
        <a class="project-index-list__title" href="{{ primary_link.url | relative_url }}">{{ project.name }}</a>
      {% endif %}
      <span>{{ project.display_date }} · {{ project.highlight }}</span>
      <p>{{ project.summary }}</p>
      <div class="project-index-list__links" aria-label="{{ project.name }} links">
{% for link in project.links %}
        {% if link.url contains "http" %}
          <a href="{{ link.url }}" target="_blank" rel="noopener">{{ link.label }}</a>
        {% else %}
          <a href="{{ link.url | relative_url }}">{{ link.label }}</a>
        {% endif %}
{% endfor %}
      </div>
    </div>
  </li>
{% endfor %}
</ul>
