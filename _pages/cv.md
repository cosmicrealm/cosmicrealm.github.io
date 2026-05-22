---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}
{% assign cv = site.data.cv_profile %}

<section class="cv-hero">
  <div class="hero-actions cv-actions">
{% for action in cv.summary.actions %}
{% if action.url contains "http" %}
    <a href="{{ action.url }}" target="_blank" rel="noopener">
{% elsif action.url contains "mailto:" %}
    <a href="{{ action.url }}">
{% else %}
    <a href="{{ action.url | relative_url }}">
{% endif %}
      <i class="{{ action.icon }}" aria-hidden="true"></i><span>{{ action.label }}</span>
    </a>
{% endfor %}
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <div>
      <p class="section-heading__eyebrow">Education</p>
      <h2>Academic Background</h2>
    </div>
  </div>
  <div class="cv-timeline cv-timeline--education">
{% for item in cv.education %}
    <article class="cv-timeline__item">
      <div class="cv-timeline__date">{{ item.period }}</div>
      <div class="cv-timeline__body">
        <div class="cv-timeline__heading">
          <div>
            <h3>{{ item.degree }}</h3>
            <p>{{ item.school }}</p>
          </div>
        </div>
      </div>
    </article>
{% endfor %}
  </div>
</section>

<section class="content-section">
  <div class="section-heading">
    <div>
      <p class="section-heading__eyebrow">Experience</p>
      <h2>Work Timeline</h2>
    </div>
  </div>
  <div class="cv-timeline">
{% for job in cv.experience %}
    <article class="cv-timeline__item">
      <div class="cv-timeline__date">{{ job.period }}</div>
      <div class="cv-timeline__body">
        <div class="cv-timeline__heading">
          <div>
            <h3>{{ job.company }}</h3>
            <p>{{ job.role }} · {{ job.location }}</p>
          </div>
        </div>
        <ul>
{% for bullet in job.bullets %}
          <li>{{ bullet }}</li>
{% endfor %}
        </ul>
        <div class="cv-chip-row" aria-label="Experience tags">
{% for tag in job.tags %}
          <span>{{ tag }}</span>
{% endfor %}
        </div>
      </div>
    </article>
{% endfor %}
  </div>
</section>
