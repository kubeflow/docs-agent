{{- define "pr-eval.name" -}}
pr-eval-{{ required "pr is required" .Values.pr }}-{{ required "headSha is required" .Values.headSha | trunc 8 }}
{{- end }}

{{- define "pr-eval.labels" -}}
app.kubernetes.io/part-of: docs-agent
app.kubernetes.io/name: pr-eval
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
docs-agent/pr: {{ .Values.pr | quote }}
{{- end }}
