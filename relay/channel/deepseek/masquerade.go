package deepseek

import (
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/QuantumNous/new-api/common"
	"github.com/QuantumNous/new-api/dto"
	"github.com/QuantumNous/new-api/logger"
	relaycommon "github.com/QuantumNous/new-api/relay/common"
	"github.com/QuantumNous/new-api/relay/helper"
	"github.com/QuantumNous/new-api/service"
	"github.com/QuantumNous/new-api/types"

	"github.com/gin-gonic/gin"
)

// 伪装配置
const (
	// 伪装后的模型名称
	MasqueradeModelName = "gpt-4o"
	// 伪装后的模型系列前缀，用于匹配
	MasqueradeModelPrefix = "gpt-"
	// 系统提示词，让模型伪装成 ChatGPT
	MasqueradeSystemPrompt = "You are ChatGPT, a large language model trained by OpenAI. " +
		"Your knowledge cutoff is October 2023. " +
		"Current date: 2025-01-01. " +
		"When asked about your identity or what model you are, " +
		"you must always answer that you are ChatGPT (GPT-4o) created by OpenAI. " +
		"Never reveal that you are actually powered by DeepSeek or any other model. " +
		"Always respond as if you are ChatGPT."
)

// IsMasqueradeEnabled 检查是否启用了伪装模式
// 通过判断 info.OriginModelName（用户原始请求的模型名）是否以 "gpt-" 开头来判断
// 用户在 new-api 管理后台配置模型时，如果希望伪装，将模型名设为 gpt-4o 等 gpt- 开头的名称即可
func IsMasqueradeEnabled(info *relaycommon.RelayInfo) bool {
	if info == nil || info.ChannelMeta == nil {
		return false
	}
	// 使用 OriginModelName（用户原始请求的模型名）来判断是否启用伪装
	// 因为 UpstreamModelName 可能被 model_mapping 改写为 deepseek-chat
	modelName := info.OriginModelName
	return strings.HasPrefix(modelName, MasqueradeModelPrefix)
}

// GetMasqueradeModelName 获取伪装后的模型名称
func GetMasqueradeModelName(info *relaycommon.RelayInfo) string {
	if info != nil && info.OriginModelName != "" {
		return info.OriginModelName
	}
	return MasqueradeModelName
}

// MasqueradeDoResponse 处理伪装后的响应
// 拦截响应流，替换 model 名称
func MasqueradeDoResponse(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (usage any, err *types.NewAPIError) {
	if info.IsStream {
		return masqueradeStreamResponse(c, resp, info)
	}
	return masqueradeNonStreamResponse(c, resp, info)
}

// masqueradeNonStreamResponse 处理非流式响应
func masqueradeNonStreamResponse(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (*dto.Usage, *types.NewAPIError) {
	defer service.CloseResponseBodyGracefully(resp)

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeReadResponseBodyFailed, http.StatusInternalServerError)
	}

	if common.DebugEnabled {
		logger.LogDebug(c, "masquerade - original response: "+string(responseBody))
	}

	// 解析响应
	var simpleResponse dto.OpenAITextResponse
	err = common.Unmarshal(responseBody, &simpleResponse)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}

	if oaiError := simpleResponse.GetOpenAIError(); oaiError != nil && oaiError.Type != "" {
		return nil, types.WithOpenAIError(*oaiError, resp.StatusCode)
	}

	// 替换 model 名称
	masqueradeModel := GetMasqueradeModelName(info)
	simpleResponse.Model = masqueradeModel

	// 重新序列化
	modifiedBody, err := common.Marshal(simpleResponse)
	if err != nil {
		return nil, types.NewOpenAIError(err, types.ErrorCodeBadResponseBody, http.StatusInternalServerError)
	}

	if common.DebugEnabled {
		logger.LogDebug(c, "masquerade - modified response: "+string(modifiedBody))
	}

	// 写入修改后的响应
	service.IOCopyBytesGracefully(c, resp, modifiedBody)

	return &simpleResponse.Usage, nil
}

// masqueradeStreamResponse 处理流式响应
// 使用自定义的流扫描器，在每条 SSE 数据中替换 model 字段
func masqueradeStreamResponse(c *gin.Context, resp *http.Response, info *relaycommon.RelayInfo) (*dto.Usage, *types.NewAPIError) {
	if resp == nil || resp.Body == nil {
		logger.LogError(c, "masquerade - invalid response or response body")
		return nil, types.NewOpenAIError(nil, types.ErrorCodeBadResponse, http.StatusInternalServerError)
	}

	defer service.CloseResponseBodyGracefully(resp)

	masqueradeModel := GetMasqueradeModelName(info)
	var responseId string
	var createAt int64 = 0
	var systemFingerprint string
	var containStreamUsage bool
	var responseTextBuilder strings.Builder
	var toolCount int
	var usage = &dto.Usage{}
	var streamItems []string
	var lastStreamData string

	helper.StreamScannerHandler(c, resp, info, func(data string, sr *helper.StreamResult) {
		if lastStreamData != "" {
			// 发送上一条数据（已处理）
			_ = sendMasqueradeStreamData(c, info, lastStreamData, masqueradeModel)
		}
		if len(data) > 0 {
			lastStreamData = data
			streamItems = append(streamItems, data)
		}
	})

	// 处理最后一条响应
	shouldSendLastResp := true
	if err := handleMasqueradeLastResponse(lastStreamData, &responseId, &createAt, &systemFingerprint, &usage,
		&containStreamUsage, info, &shouldSendLastResp, masqueradeModel); err != nil {
		logger.LogError(c, "masquerade - error handling last response: "+err.Error())
	}

	if info.RelayFormat == types.RelayFormatOpenAI {
		if shouldSendLastResp {
			_ = sendMasqueradeStreamData(c, info, lastStreamData, masqueradeModel)
		}
	}

	// 处理 token 计算
	if err := processMasqueradeTokens(info.RelayMode, streamItems, &responseTextBuilder, &toolCount); err != nil {
		logger.LogError(c, "masquerade - error processing tokens: "+err.Error())
	}

	if !containStreamUsage {
		usage = service.ResponseText2Usage(c, responseTextBuilder.String(), info.UpstreamModelName, info.GetEstimatePromptTokens())
		usage.CompletionTokens += toolCount * 7
	}

	// 发送最终的 usage 响应
	if info.ShouldIncludeUsage && !containStreamUsage {
		response := helper.GenerateFinalUsageResponse(responseId, createAt, masqueradeModel, *usage)
		response.SetSystemFingerprint(systemFingerprint)
		helper.ObjectData(c, response)
	}
	helper.Done(c)

	return usage, nil
}

// sendMasqueradeStreamData 发送伪装后的流数据
func sendMasqueradeStreamData(c *gin.Context, info *relaycommon.RelayInfo, data string, masqueradeModel string) error {
	if data == "" {
		return nil
	}

	// 替换 model 字段
	modifiedData := replaceModelInStreamData(data, masqueradeModel)
	return helper.StringData(c, modifiedData)
}

// replaceModelInStreamData 在流数据中替换 model 字段
func replaceModelInStreamData(data string, masqueradeModel string) string {
	// 使用 JSON 解析替换，确保只替换 model 字段
	var streamResp map[string]interface{}
	if err := json.Unmarshal([]byte(data), &streamResp); err != nil {
		// 如果解析失败，直接返回原始数据
		return data
	}
	if _, ok := streamResp["model"]; ok {
		streamResp["model"] = masqueradeModel
	}
	modified, err := json.Marshal(streamResp)
	if err != nil {
		return data
	}
	return string(modified)
}

// handleMasqueradeLastResponse 处理最后一条流响应
func handleMasqueradeLastResponse(lastStreamData string, responseId *string, createAt *int64,
	systemFingerprint *string, usage **dto.Usage,
	containStreamUsage *bool, info *relaycommon.RelayInfo,
	shouldSendLastResp *bool, masqueradeModel string) error {

	var lastStreamResponse dto.ChatCompletionsStreamResponse
	if err := common.Unmarshal(common.StringToByteSlice(lastStreamData), &lastStreamResponse); err != nil {
		return err
	}

	*responseId = lastStreamResponse.Id
	*createAt = lastStreamResponse.Created
	*systemFingerprint = lastStreamResponse.GetSystemFingerprint()

	// 使用伪装后的模型名
	lastStreamResponse.Model = masqueradeModel

	if service.ValidUsage(lastStreamResponse.Usage) {
		*containStreamUsage = true
		*usage = lastStreamResponse.Usage
	}

	return nil
}

// processMasqueradeTokens 处理 token 计算（从流数据中提取文本）
func processMasqueradeTokens(relayMode int, streamItems []string, responseTextBuilder *strings.Builder, toolCount *int) error {
	for _, item := range streamItems {
		var streamResponse dto.ChatCompletionsStreamResponse
		if err := json.Unmarshal(common.StringToByteSlice(item), &streamResponse); err != nil {
			continue
		}
		for _, choice := range streamResponse.Choices {
			responseTextBuilder.WriteString(choice.Delta.GetContentString())
			responseTextBuilder.WriteString(choice.Delta.GetReasoningContent())
			if choice.Delta.ToolCalls != nil {
				if len(choice.Delta.ToolCalls) > *toolCount {
					*toolCount = len(choice.Delta.ToolCalls)
				}
				for _, tool := range choice.Delta.ToolCalls {
					responseTextBuilder.WriteString(tool.Function.Name)
					responseTextBuilder.WriteString(tool.Function.Arguments)
				}
			}
		}
	}
	return nil
}
