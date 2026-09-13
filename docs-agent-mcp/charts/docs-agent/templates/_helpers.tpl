{{- define "docs-agent.labels" -}}
app.kubernetes.io/part-of: docs-agent
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version | replace "+" "_" }}
{{- end }}

{{- define "docs-agent.mcpName" -}}
mcp-kubeflow-docs
{{- end }}

{{- define "docs-agent.mcpSelectorLabels" -}}
app: {{ include "docs-agent.mcpName" . }}
app.kubernetes.io/name: {{ include "docs-agent.mcpName" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
