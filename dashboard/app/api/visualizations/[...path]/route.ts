import { NextRequest, NextResponse } from 'next/server'
import { readFileSync, existsSync } from 'fs'
import { join } from 'path'

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    const filePath = join(process.cwd(), '..', 'results', 'plots', ...params.path)
    
    if (!existsSync(filePath)) {
      return NextResponse.json({ error: 'Visualization not found' }, { status: 404 })
    }

    const fileBuffer = readFileSync(filePath)
    const fileExtension = filePath.split('.').pop()?.toLowerCase()
    
    let contentType = 'application/octet-stream'
    if (fileExtension === 'png') contentType = 'image/png'
    else if (fileExtension === 'jpg' || fileExtension === 'jpeg') contentType = 'image/jpeg'
    else if (fileExtension === 'svg') contentType = 'image/svg+xml'
    else if (fileExtension === 'pdf') contentType = 'application/pdf'
    
    return new NextResponse(fileBuffer, {
      headers: {
        'Content-Type': contentType,
        'Cache-Control': 'public, max-age=3600',
      },
    })
  } catch (error) {
    console.error('Error serving visualization:', error)
    return NextResponse.json({ error: 'Failed to load visualization' }, { status: 500 })
  }
}