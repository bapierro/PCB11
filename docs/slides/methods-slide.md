---
theme: seriph
title: "PCB11 Methods"
---

# Methods Pipeline

<div class="method-diagram">
  <div class="method-row">
    <div class="method-box">
      <strong>Input Data</strong>
      <small>SYNS images<br>MEG (20 participants)<br>Behavioral scene labels</small>
    </div>
    <div class="method-arrow">&rarr;</div>
    <div class="method-box">
      <strong>Representations</strong>
      <small>MEG RDMs over time<br>ANN features<br>Vision models</small>
    </div>
  </div>
  <div class="method-row">
    <div class="method-box method-accent">
      <strong>RSA Comparison</strong>
      <small>Spearman correlation<br>ANN-layer RDM vs MEG-time RDM</small>
    </div>
    <div class="method-arrow">&rarr;</div>
    <div class="method-box">
      <strong>Main Test</strong>
      <small>Do deeper ANN layers match later MEG dynamics?<br>Does behavior training improve alignment?</small>
    </div>
  </div>
</div>

<div class="method-note">
Goal: test whether behavior-trained ANNs better capture the brain's temporal hierarchy in scene processing.
</div>

<style>
h1 {
  font-size: 2.2rem;
  margin-bottom: 0.8rem;
}
.method-diagram {
  display: flex;
  flex-direction: column;
  gap: 0.45rem;
}
.method-row {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 0.6rem;
}
.method-box {
  width: 230px;
  min-height: 95px;
  border: 2px solid rgba(255, 255, 255, 0.62);
  border-radius: 12px;
  padding: 0.5rem 0.55rem;
  text-align: center;
  line-height: 1.15;
  background: rgba(255, 255, 255, 0.04);
}
.method-box strong {
  display: block;
  font-size: 0.92rem;
  margin-bottom: 0.28rem;
}
.method-box small {
  font-size: 0.69rem;
  color: rgba(255, 255, 255, 0.9);
}
.method-accent {
  border-color: #8ce0b0;
  background: rgba(140, 224, 176, 0.14);
}
.method-arrow {
  font-size: 1.25rem;
  font-weight: 700;
  opacity: 0.92;
}
.method-note {
  margin-top: 0.55rem;
  text-align: center;
  font-size: 0.78rem;
  color: rgba(255, 255, 255, 0.9);
}
</style>
