{{- define "qwen-runtime.labels" -}}
app.kubernetes.io/part-of: docs-agent
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/name: qwen-runtime
app.kubernetes.io/instance: {{ .Release.Name }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}
