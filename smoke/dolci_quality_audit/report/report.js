"use strict";
const $ = id => document.getElementById(id);
const names = {major_defect:"Major defect",minor_defect:"Minor defect",uncertain:"Uncertain",no_defect_found:"No defect found"};
const dims = {prompt_quality:"Prompt quality",answer_correctness:"Answer correctness",reasoning_quality:"Reasoning quality",instruction_following:"Instruction following",parsing_integrity:"Parsing / generation",training_suitability:"Training suitability"};
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[ch]));
let data = null, lookup = new Map(), filtered = [], page = 0;
const pageSize = 20;
$("printButton").addEventListener("click", () => window.print());
const sourceRows = Array.from(document.querySelectorAll("#sourceTable tbody tr"));
function sourceTable() {
 const query = $("sourceSearch").value.toLowerCase();
 const sort = $("sourceSort").value;
 sourceRows.sort((a,b) => sort === "name" ? a.dataset.source.localeCompare(b.dataset.source) : Number(b.dataset[sort])-Number(a.dataset[sort]));
 for (const row of sourceRows) {row.hidden = !row.dataset.source.toLowerCase().includes(query);$("sourceTable").tBodies[0].appendChild(row);}
}
$("sourceSearch").addEventListener("input", sourceTable);
$("sourceSort").addEventListener("change", sourceTable);
function csvCell(value) {
 let str = String(value ?? "");
 if (/^[=+@\-\t\r]/.test(str)) str = "'" + str;
 return '"'+str.replaceAll('"','""')+'"';
}
$("downloadSources").addEventListener("click", () => {
 if (!data) return;
 const header = ["source","population","sampled","population_share_pct","major_count","minor_count","uncertain_count","no_defect_count","major_rate_pct","major_ci_low_pct","major_ci_high_pct","major_contribution_pp"];
 const rows = data.analysis.sources.map(s => [s.source,s.population,s.sampled,100*s.weight,s.counts.major_defect,s.counts.minor_defect,s.counts.uncertain,s.counts.no_defect_found,100*s.major_rate,100*s.major_interval[0],100*s.major_interval[1],s.major_contribution_pp]);
 const url = URL.createObjectURL(new Blob([[header,...rows].map(r=>r.map(csvCell).join(",")).join("\r\n")],{type:"text/csv;charset=utf-8"}));
 const anchor = document.createElement("a");anchor.href=url;anchor.download="dolci-source-quality-summary.csv";anchor.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
function applyFilters() {
 if (!data) return;
 const source=$("reviewSource").value, outcome=$("reviewOutcome").value, dimension=$("reviewDimension").value, query=$("reviewSearch").value.toLowerCase().trim();
 filtered=data.reviews.filter(r => (!source||r.source===source)&&(!outcome||r.overall===outcome)&&(!dimension||["minor_issue","major_issue","uncertain"].includes(r.dimensions[dimension]))&&(!query||r.searchText.includes(query)));
 page=0; renderReviews();
}
function renderReviews() {
 const count=filtered.length, totalPages=Math.max(1,Math.ceil(count/pageSize));page=Math.min(page,totalPages-1);
 $("resultCount").textContent=count.toLocaleString()+" matching reviews · "+data.reviews.length.toLocaleString()+" total";
 $("pageNumber").textContent="Page "+(page+1)+" of "+totalPages;
 $("previousPage").disabled=page===0;$("nextPage").disabled=page+1>=totalPages;
 const rows=filtered.slice(page*pageSize,(page+1)*pageSize);
 $("reviewCards").innerHTML=rows.length?rows.map(r=>`<article class="review-card"><div class="review-card-head"><span class="review-source">${esc(r.source)}</span><span class="badge ${r.overall}">${names[r.overall]}</span></div><p>${esc(r.summary)}</p><div class="prompt-preview">${esc(r.preview.slice(0,280))}${r.preview.length>280?"…":""}</div><div class="review-card-bottom"><code>${esc(r.audit_id)}</code><button class="text-button" data-open-record="${esc(r.audit_id)}">Open full record ↗</button></div></article>`).join(""):'<div class="empty">No records match these filters.</div>';
}
for(const id of ["reviewSource","reviewOutcome","reviewDimension"]) $(id).addEventListener("change",applyFilters);
let searchTimer;
$("reviewSearch").addEventListener("input",()=>{clearTimeout(searchTimer);searchTimer=setTimeout(applyFilters,180);});
$("previousPage").addEventListener("click",()=>{page--;renderReviews();$("resultCount").scrollIntoView({block:"start"});});
$("nextPage").addEventListener("click",()=>{page++;renderReviews();$("resultCount").scrollIntoView({block:"start"});});
$("resetFilters").addEventListener("click",()=>{for(const id of ["reviewSource","reviewOutcome","reviewDimension","reviewSearch"])$(id).value="";applyFilters();});
function openRecord(id) {
 const r=lookup.get(id);if(!r)return;
 $("dialogTitle").textContent="Record evidence";
 $("dialogBody").innerHTML=`<div class="record-meta"><strong>${esc(r.source)}</strong> · <span class="badge ${r.overall}">${names[r.overall]}</span><br>Audit ID: ${esc(r.audit_id)}<br>Original ID: ${esc(r.prompt_id)}<br>${esc(r.input_shard)} · zero-based row ${r.input_row_zero_based}<br>Record SHA-256: ${esc(r.record_sha256)}<br>Reviewer: ${esc(r.model)} · ${r.record.messages.length} messages</div><p class="record-summary">${esc(r.summary)}</p><div class="dimension-pills">${Object.entries(r.dimensions).map(([k,v])=>`<span class="dimension-pill ${v}">${dims[k]}: ${esc(v.replaceAll("_"," "))}</span>`).join("")}</div>${r.uncertainty_reason?`<div class="callout"><strong>Uncertainty</strong><br>${esc(r.uncertainty_reason)}</div>`:""}<h3>Reviewer findings</h3>${r.issues.length?r.issues.map(i=>`<article class="issue"><h4>${dims[i.dimension]} · ${esc(i.severity)}</h4><blockquote>${esc(i.evidence)}</blockquote><p>${esc(i.explanation)}</p></article>`).join(""):'<p>No specific defect was identified by the reviewer.</p>'}<h3>Complete original conversation</h3>${r.record.messages.map((m,i)=>`<article class="message"><div class="message-role">Turn ${i} · ${esc(m.role)}</div><details class="record-text" open><summary>Original content · ${m.content.length.toLocaleString()} characters</summary><pre>${esc(m.content)}</pre></details>${m.role==="assistant"?`<details class="record-text"><summary>Audit parsing views</summary><h4>Reasoning</h4><pre>${esc(m.content.slice(r.parsed[i].reasoning_start,r.parsed[i].reasoning_end))}</pre><h4>Answer</h4><pre>${esc(m.content.slice(r.parsed[i].answer_start))}</pre>${r.parsed[i].parser_notes.length?`<p>Adapter notes: ${esc(r.parsed[i].parser_notes.join("; "))}</p>`:""}</details>`:""}</article>`).join("")}`;
 $("recordDialog").showModal();$("recordDialog").scrollTop=0;$("closeDialog").focus();
}
$("closeDialog").addEventListener("click",()=>$("recordDialog").close());
$("recordDialog").addEventListener("click",e=>{if(e.target===$("recordDialog")){const r=$("recordDialog").getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)$("recordDialog").close();}});
document.addEventListener("click",event=>{
 const record=event.target.closest("[data-open-record]");if(record){openRecord(record.dataset.openRecord);return;}
 const source=event.target.closest("[data-open-source]");if(source&&data){$("reviewSource").value=source.dataset.openSource;$("reviewOutcome").value="";$("reviewDimension").value="";$("reviewSearch").value="";applyFilters();$("explorer").scrollIntoView({block:"start"});}
});
if("IntersectionObserver" in window){const observer=new IntersectionObserver(entries=>{for(const entry of entries){if(entry.isIntersecting){for(const a of document.querySelectorAll("nav a"))a.classList.toggle("active",a.getAttribute("href")==="#"+entry.target.id);}}},{rootMargin:"-10% 0px -70% 0px"});document.querySelectorAll("main>section").forEach(s=>observer.observe(s));}
(async()=>{
 try{
  const bytes=Uint8Array.from(atob($("packedData").textContent.trim()),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  data=JSON.parse(await new Response(stream).text());
  for(const r of data.reviews){lookup.set(r.audit_id,r);r.preview=r.record.messages.find(m=>m.role==="user")?.content||"";r.searchText=[r.audit_id,r.source,...r.record.messages.map(m=>m.content),r.summary,r.uncertainty_reason,...r.issues.flatMap(i=>[i.evidence,i.explanation])].join("\n").toLowerCase();}
  for(const source of data.analysis.sources.map(s=>s.source).sort()){const option=document.createElement("option");option.value=source;option.textContent=source;$("reviewSource").appendChild(option);}
  $("loading").textContent="All 3,200 reviews loaded locally. No network connection is used.";
  applyFilters();window.auditReady=true;window.auditRecordCount=data.reviews.length;
 }catch(error){$("loading").textContent="The interactive explorer needs a browser supporting gzip DecompressionStream. The static analysis and complete source table remain available. Error: "+error.message;window.auditLoadError=error.message;}
})();
