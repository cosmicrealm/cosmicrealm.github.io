---
layout: archive
title: "Projects"
permalink: /projects/
author_profile: true
---

<ul class="index-list project-index-list">
{% for project in site.data.projects %}
  <li>
    <div class="project-index-list__title">
      {% assign primary_link = project.links | first %}
      {% if primary_link.url contains "http" %}
        <a href="{{ primary_link.url }}" target="_blank" rel="noopener">{{ project.name }}</a>
      {% else %}
        <a href="{{ primary_link.url | relative_url }}">{{ project.name }}</a>
      {% endif %}
    </div>
    <div class="project-index-list__body">
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
