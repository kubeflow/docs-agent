{{- define "llm-runtime.labels" -}}
app.kubernetes.io/part-of: docs-agent
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/name: llm-runtime
app.kubernetes.io/instance: {{ .Release.Name }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}

{{/* The catalog profile selected by .Values.model. */}}
{{- define "llm-runtime.profile" -}}
{{- $p := get .Values.catalog .Values.model }}
{{- if not $p }}
{{- fail (printf "model %q is not in catalog (have: %s)" .Values.model (keys .Values.catalog | sortAlpha | join ", ")) }}
{{- end }}
{{- toYaml $p }}
{{- end }}

{{/* vLLM server args for the selected profile. */}}
{{- define "llm-runtime.args" -}}
{{- $p := include "llm-runtime.profile" . | fromYaml -}}
- --model={{ $p.id }}
{{- with $p.revision }}
- --revision={{ . }}
{{- end }}
- --served-model-name={{ .Values.servedModelName }}
- --port={{ .Values.service.targetPort }}
- --max-model-len={{ $p.maxModelLen }}
- --gpu-memory-utilization={{ $p.gpuMemoryUtilization }}
{{- with $p.quantization }}
- --quantization={{ . }}
{{- end }}
- --enable-auto-tool-choice
- --tool-call-parser={{ $p.toolCallParser }}
{{- with $p.reasoningParser }}
- --reasoning-parser={{ . }}
{{- end }}
{{- if kindIs "bool" $p.thinkingDefault }}
- {{ printf "--default-chat-template-kwargs={\"enable_thinking\": %t}" $p.thinkingDefault | quote }}
{{- end }}
{{- range $p.extraArgs }}
- {{ . | quote }}
{{- end }}
{{- end }}

{{/* requests/limits for the selected profile; empty values are omitted. */}}
{{- define "llm-runtime.resources" -}}
{{- $r := (include "llm-runtime.profile" . | fromYaml).resources -}}
{{- /* KServe fills any missing value with its cluster default (cpu 1, memory 2Gi), which rejects or starves vLLM. */ -}}
requests:
  cpu: {{ required "catalog profile needs resources.cpu.request" $r.cpu.request | quote }}
  memory: {{ required "catalog profile needs resources.memory.request" $r.memory.request }}
  nvidia.com/gpu: "1"
limits:
  cpu: {{ required "catalog profile needs resources.cpu.limit" $r.cpu.limit | quote }}
  memory: {{ required "catalog profile needs resources.memory.limit" $r.memory.limit }}
  nvidia.com/gpu: "1"
{{- end }}
