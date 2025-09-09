import { NextRequest, NextResponse } from 'next/server'
import { llmService } from '@/lib/llmService'

export async function POST(request: NextRequest) {
  try {
    const { message, userRole, userType, userId, browserContext } = await request.json()

    if (!message || !userRole) {
      return NextResponse.json(
        { error: 'Message and user role are required' },
        { status: 400 }
      )
    }

    // Use enhanced chat method with browser context if available
    const response = browserContext 
      ? await llmService.chatWithContext(message, userRole, userType, userId, browserContext)
      : await llmService.chat(message, userRole, userType, userId)

    return NextResponse.json({
      success: true,
      data: response
    })
  } catch (error) {
    console.error('Chat API Error:', error)
    return NextResponse.json(
      { error: 'Failed to process chat request' },
      { status: 500 }
    )
  }
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const userRole = searchParams.get('userRole') as 'user' | 'admin'
    const userType = searchParams.get('userType') as 'Driver' | 'Merchant' | undefined
    const userId = searchParams.get('userId') || undefined

    const suggestions = await llmService.getSmartSuggestions(userRole, userType, userId)

    return NextResponse.json({
      success: true,
      data: { suggestions }
    })
  } catch (error) {
    console.error('Chat Suggestions API Error:', error)
    return NextResponse.json(
      { error: 'Failed to get suggestions' },
      { status: 500 }
    )
  }
}